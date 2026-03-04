import optuna
from transformers import TrainingArguments, Trainer

def optuna_hp_space(trial: optuna.trial.Trial):
    """Espacio de búsqueda para fine-tuning. Retorna un dict mutuamente
    compatible con TrainingArguments."""
    return {
        # LR log-uniform entre 5e-6 y 5e-5
        "learning_rate": trial.suggest_float("learning_rate",
                                             5e-6, 5e-5,
                                             log=True),
        # Warm-up proporcional a total_steps
        "warmup_ratio": trial.suggest_float("warmup_ratio",
                                            0.0, 0.2),
        # Weight decay suele beneficiar a modelos grandes
        "weight_decay": trial.suggest_float("weight_decay",
                                            0.0, 0.1),
        # Número de épocas: valor entero discreto
        "num_train_epochs": trial.suggest_int("num_train_epochs",
                                              2, 6),
        # Tamaño de lote; debes ser múltiplo de 2 si usas grad. acumulación
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32]),
        # Estrategia de scheduler
        "lr_scheduler_type": trial.suggest_categorical(
            "lr_scheduler_type",
            ["linear", "cosine", "cosine_with_restarts",
             "polynomial", "constant"]),
        # Gradientes acumulados para emular batch grande
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", [1, 2, 4]),
        # Dropout de atención y de hidden (solo si tu modelo lo expone)
        "attention_dropout": trial.suggest_float(
            "attention_dropout", 0.0, 0.3),
        "hidden_dropout": trial.suggest_float(
            "hidden_dropout", 0.0, 0.3),
    }
