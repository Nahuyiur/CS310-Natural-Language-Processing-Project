import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
import os
import csv

def train_val_test_pipeline(
    model_name: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    dataset_class,
    num_labels: int = 2,
    max_length: int = 128,
    batch_size: int = 8,
    epochs: int = 3,
    learning_rate: float = 3e-5,
    save_best_model_path: str = "None",
    log_interval: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    isFT: bool=False,

):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    # Load datasets
    train_dataset = dataset_class(train_csv, tokenizer, max_length=max_length)
    val_dataset = dataset_class(val_csv, tokenizer, max_length=max_length)
    test_dataset = dataset_class(test_csv, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    best_val_f1 = 0.0
    loss_list = []# 用于记录训练损失
    step_list = []
    global_step = 0#全局step

    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        running_loss = 0
        for step, batch in enumerate(train_loader):
            global_step += 1#全局step
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            running_loss += loss.item()

            if (step + 1) % log_interval == 0:
                print(f"[Epoch {epoch+1}] Step {step+1}/{len(train_loader)} - Batch Loss: {running_loss / log_interval:.4f}", flush=True)
                loss_list.append(running_loss / log_interval)
                step_list.append(global_step)
                running_loss = 0

        print(f"[Epoch {epoch + 1}] Train Loss: {total_loss / len(train_loader):.4f}", flush=True)

        # Validation
        val_labels, val_preds, val_probs, val_domains = evaluate_model(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        print(f"[Epoch {epoch + 1}] Validation F1: {val_f1:.4f}", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_best_model_path)
            print(f"New best model saved to {save_best_model_path}", flush=True)
    # 在所有 epoch 训练完成后添加（保存 loss 曲线图）：
    def save_loss_data(step_list, loss_list, filename="xlm-roberta-base-en-mix.csv"):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Step', 'Loss'])  # 写入标题行
            for step, loss in zip(step_list, loss_list):
                writer.writerow([step, loss])
            
# 在训练循环中调用
    save_loss_data(step_list, loss_list)
    def load_and_plot_loss_data(csv_file="training_loss.csv", output_html="training_loss.html"):
        step_list = []
        loss_list = []
        
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过标题行
            for row in reader:
                step_list.append(int(row[0]))
                loss_list.append(float(row[1]))
        
        # 创建图表
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=step_list,
            y=loss_list,
            mode='lines+markers',
            name='Train Loss',
            line=dict(color='royalblue', width=2),
            marker=dict(size=4),
            hovertemplate='Step %{x}<br>Loss: %{y:.4f}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text='Training Loss vs Steps', x=0.5, font=dict(size=24, family='Arial')),
            xaxis=dict(
                title='Step',
                tickfont=dict(size=14),
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                title='Loss',
                tickfont=dict(size=14),
                gridcolor='lightgrey'
            ),
            plot_bgcolor='white',
            hovermode='x unified',
            legend=dict(
                font=dict(size=14),
                bgcolor='rgba(255,255,255,0)',
                borderwidth=0,
                x=0.01,
                y=0.99
            )
        )
        
        fig.write_html(output_html)
        return fig

# 使用示例
    #load_and_plot_loss_data()
    # fig.write_image("training_loss.png")  # PNG 需要安装 kaleido 库：`pip install -U kaleido`
    # Test
    print("Final Evaluation on Test Set", flush=True)
    model.load_state_dict(torch.load(save_best_model_path))
    model.to(device)
    test_labels, test_preds, test_probs, test_domains = evaluate_model(model, test_loader, device)

    print("[Overall Test Metrics]", flush=True)
    print_metrics(test_labels, test_preds, test_probs)

    print("[Domain-wise Test Metrics]", flush=True)
    report_by_domain(test_labels, test_preds, test_probs, test_domains)


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_probs, all_labels, all_domains = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            domains = batch["domain"]

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_domains.extend(domains)

    return all_labels, all_preds, all_probs, all_domains


def print_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except:
        auroc = float("nan")

    print(f"Accuracy:  {acc:.4f}", flush=True)
    print(f"Precision: {prec:.4f}", flush=True)
    print(f"Recall:    {rec:.4f}", flush=True)
    print(f"F1 Score:  {f1:.4f}", flush=True)
    print(f"AUROC:     {auroc:.4f}", flush=True)


def report_by_domain(y_true, y_pred, y_prob, domains):
    grouped = defaultdict(lambda: {"y": [], "p": [], "prob": []})
    for yt, yp, pr, dom in zip(y_true, y_pred, y_prob, domains):
        grouped[dom]["y"].append(yt)
        grouped[dom]["p"].append(yp)
        grouped[dom]["prob"].append(pr)

    for dom, data in grouped.items():
        print(f"\nDomain: {dom}", flush=True)
        print_metrics(data["y"], data["p"], data["prob"])
