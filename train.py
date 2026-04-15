import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os
import pandas as pd

# ── Device ─────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ── Dataset ─────────────────────────────────────────────────────────────
print("Loading IMDb dataset...")
dataset = load_dataset("imdb")

# use full test set this time for accurate evaluation
train_data = dataset["train"].shuffle(seed=42).select(range(15000))
test_data  = dataset["test"]  # full 25k test set

# ── Tokenizer ────────────────────────────────────────────────────────────
print("Loading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256        # back to 256 — captures more context
    )

print("Tokenizing...")
train_tokenized = train_data.map(tokenize, batched=True, batch_size=512)
test_tokenized  = test_data.map(tokenize,  batched=True, batch_size=512)

train_tokenized = train_tokenized.rename_column("label", "labels")
test_tokenized  = test_tokenized.rename_column("label", "labels")

train_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_tokenized.set_format("torch",  columns=["input_ids", "attention_mask", "labels"])

# ── Model ─────────────────────────────────────────────────────────────────
print("Loading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
model.to(device)

# ── Custom training loop — no Trainer, full control ───────────────────────
EPOCHS      = 3
BATCH_SIZE  = 8
LR          = 3e-5           # higher learning rate — key change

train_loader = DataLoader(train_tokenized, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_tokenized,  batch_size=32)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps   = len(train_loader) * EPOCHS
warmup_steps  = int(0.06 * total_steps)
scheduler     = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"\nTraining for {EPOCHS} epochs, {len(train_loader)} steps/epoch")
print(f"Warmup steps: {warmup_steps}, Total steps: {total_steps}\n")

best_f1 = 0.0

for epoch in range(EPOCHS):
    # ── Train ──────────────────────────────────────────────────────────
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       labels=labels)
        loss = outputs.loss

        loss.backward()

        # gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if (step + 1) % 200 == 0:
            avg = total_loss / (step + 1)
            lr  = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {avg:.4f} | LR: {lr:.2e}")

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} complete — avg loss: {avg_loss:.4f}")

    # ── Evaluate after each epoch ──────────────────────────────────────
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    f1  = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1} — F1: {f1:.4f} | Accuracy: {acc:.4f}\n")

    if f1 > best_f1:
        best_f1 = f1
        # save best model directly
        os.makedirs("models/distilbert-final", exist_ok=True)
        model.save_pretrained("models/distilbert-final")
        tokenizer.save_pretrained("models/distilbert-final")
        print(f"  New best model saved — F1: {best_f1:.4f}\n")

# ── Final results ──────────────────────────────────────────────────────────
print("\n=== FINAL RESULTS ===")
print(f"Best F1:       {best_f1:.4f}")
print(f"Baseline F1:   0.8830")
imp     = best_f1 - 0.8830
imp_pct = (imp / 0.8830) * 100
if imp > 0:
    print(f"Improvement:   +{imp:.4f}  (+{imp_pct:.1f}%)")
else:
    print(f"Difference:    {imp:.4f}  ({imp_pct:.1f}%)")

results_df = pd.DataFrame({
    "Model":    ["TF-IDF + Logistic Regression", "Fine-tuned DistilBERT"],
    "F1 Score": [0.8830, round(best_f1, 4)],
    "Accuracy": ["88.00%", "see above"]
})
results_df.to_csv("data/final_results.csv", index=False)
print("Results saved to data/final_results.csv")
print("\nDay 2 complete!")