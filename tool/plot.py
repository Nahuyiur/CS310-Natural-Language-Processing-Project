import plotly.graph_objects as go
import os
import csv
def load_and_plot_multiple_loss_data(csv_files, output_html="training_loss.html", labels=None):
    fig = go.Figure()

    for i, csv_file in enumerate(csv_files):
        step_list = []
        loss_list = []

        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过标题行
            for row in reader:
                step_list.append(int(row[0]))
                loss_list.append(float(row[1]))
        
        label = labels[i] if labels and i < len(labels) else f"Train Loss {i+1}"

        fig.add_trace(go.Scatter(
            x=step_list,
            y=loss_list,
            mode='lines+markers',
            name=label,
            line=dict(width=2),
            marker=dict(size=4),
            hovertemplate='Step %{x}<br>Loss: %{y:.4f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text='Dataset: Zh-wiki',  # 用 <br> 换行
            x=0.5,  # 标题居中
            y=0.95,  # 调整垂直位置（可选）
            font=dict(size=24, family='Arial')
        ),
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
            font=dict(size=20),
            bgcolor='rgba(255,255,255,0)',
            borderwidth=0,
            x=0.7,         # 放到图的右边
            xanchor='left', # 图例左对齐
            y=1,
            yanchor='top'   # 图例顶端对齐
)
    )

    fig.write_html(output_html)
    return fig

csv_files = ["bert-base-chinese-zh-wiki.csv","bert-base-multilingual-cased-zh-wiki.csv","xlm-roberta-base-zh-wiki.csv"]
labels = ["bert-base-chinese","bert-base-multilingual-cased","xlm-roberta-base"]
load_and_plot_multiple_loss_data(csv_files, output_html="all_training_loss.html", labels=labels)