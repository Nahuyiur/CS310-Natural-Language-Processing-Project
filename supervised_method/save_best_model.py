from text_classification_dataset import *
from train_val_test_pipeline import *


train_val_test_pipeline(
    model_name='/lab/haoq_lab/cse12310520/NLP/proj/pretrained_model/xlm-roberta-base',
    train_csv='/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/eng_mix/eng_train.csv',
    val_csv='/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/eng_mix/eng_val.csv',
    test_csv='/lab/haoq_lab/cse12310520/NLP/proj/prepared_data/eng_mix/eng_test.csv',
    dataset_class=TextClassificationDataset,
    batch_size=16,
    epochs=10,
    log_interval=50,
    learning_rate=1e-7,
    save_best_model_path='/lab/haoq_lab/cse12310520/NLP/proj/checkpoints/xlm-roberta-base-eng_mix.pt'
)
