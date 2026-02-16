import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoConfig, AutoTokenizer

class MultiTaskBERT(nn.Module):
    """
    Мультизадачная модель BERT для классификации категорий и тональности
    Тональность определяется для каждой категории отдельно
    """
    
    def __init__(self, model_name, num_categories, num_sentiments, dropout_rate=0.3):
        super(MultiTaskBERT, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Размер скрытого слоя BERT
        hidden_size = self.config.hidden_size
        
        # Общий слой для извлечения признаков
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier_shared = nn.Linear(hidden_size, hidden_size // 2)
        
        # Единая головка для классификации категорий и тональности
        # Выход: num_categories * 4 (каждая категория имеет 4 тональности)
        self.unified_classifier = nn.Linear(hidden_size // 2, num_categories * 4)
        
        self.num_categories = num_categories
        self.num_sentiments = num_sentiments
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов классификаторов"""
        for module in [self.classifier_shared, self.unified_classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids, attention_mask):
        """
        Прямой проход модели
        
        Args:
            input_ids: Токенизированные входные данные
            attention_mask: Маска внимания
            
        Returns:
            category_logits: Логиты для категорий
            sentiment_logits: Логиты для тональности каждой категории (batch_size, num_categories, num_sentiments)
        """
        # Получаем выходы BERT с gradient checkpointing для экономии памяти
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        
        # Используем [CLS] токен для классификации
        pooled_output = outputs.pooler_output
        
        # Общий слой признаков
        shared_features = self.dropout(pooled_output)
        shared_features = F.relu(self.classifier_shared(shared_features))
        shared_features = self.dropout(shared_features)
        
        # Единая классификация категорий и тональности
        unified_logits = self.unified_classifier(shared_features)
        # Преобразуем в форму (batch_size, num_categories, 4)
        unified_logits = unified_logits.view(-1, self.num_categories, 4)
        
        # Извлекаем логиты для категорий (максимум по тональности для каждой категории)
        category_logits = unified_logits.max(dim=2)[0]  # (batch_size, num_categories)
        
        # Логиты тональности остаются как есть
        sentiment_logits = unified_logits  # (batch_size, num_categories, 4)
        
        return category_logits, sentiment_logits

class FocalLoss(nn.Module):
    """
    Focal Loss для решения проблемы дисбаланса классов
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultiTaskLoss(nn.Module):
    """
    Комбинированная функция потерь для мультизадачного обучения с весами для дисбаланса классов
    """
    def __init__(self, category_weight=1.0, sentiment_weight=1.0, use_focal_loss=True, 
                 category_sentiment_weights=None):
        super(MultiTaskLoss, self).__init__()
        self.category_weight = category_weight
        self.sentiment_weight = sentiment_weight
        
        if use_focal_loss:
            self.category_criterion = nn.BCEWithLogitsLoss()
            self.sentiment_criterion = FocalLoss()
        else:
            self.category_criterion = nn.BCEWithLogitsLoss()
            self.sentiment_criterion = nn.CrossEntropyLoss()
        
        # Веса для пар (категория, тональность)
        self.category_sentiment_weights = category_sentiment_weights
    
    def forward(self, category_logits, sentiment_logits, category_targets, sentiment_targets):
        """
        Вычисляет комбинированную функцию потерь с учетом весов для дисбаланса классов
        
        Args:
            category_logits: Логиты для категорий
            sentiment_logits: Логиты для тональности (batch_size, num_categories, 4)
            category_targets: Целевые значения категорий
            sentiment_targets: Целевые значения тональности для каждой категории (batch_size, num_categories, 4)
            
        Returns:
            total_loss: Общая функция потерь
            category_loss: Потери по категориям
            sentiment_loss: Потери по тональности
        """
        # Потери по категориям (мультилейбл) - используем средние веса по тональности
        if self.category_sentiment_weights is not None:
            # Вычисляем средние веса для каждой категории по всем тональностям
            category_weights = np.mean(self.category_sentiment_weights, axis=1)
            category_weights_tensor = torch.tensor(category_weights, device=category_logits.device)
            category_loss = F.binary_cross_entropy_with_logits(
                category_logits, category_targets, 
                weight=category_weights_tensor, reduction='mean'
            )
        else:
            category_loss = self.category_criterion(category_logits, category_targets)
        
        # Потери по тональности для каждой категории с учетом присутствия
        sentiment_losses = []
        
        num_categories = sentiment_logits.size(1)  # Получаем количество категорий из размерности тензора
        for i in range(num_categories):
            # Берем логиты для i-й категории
            cat_sentiment_logits = sentiment_logits[:, i, :]  # (batch_size, 4)
            cat_sentiment_targets = sentiment_targets[:, i, :]  # (batch_size, 4)
            
            # Вычисляем потери только для присутствующих категорий
            present_mask = category_targets[:, i] == 1
            if present_mask.sum() > 0:
                present_logits = cat_sentiment_logits[present_mask]
                present_targets = cat_sentiment_targets[present_mask]
                
                # Получаем индексы классов
                target_indices = torch.argmax(present_targets, dim=1)
                
                # Вычисляем потери с весами для пар (категория, тональность)
                if self.category_sentiment_weights is not None:
                    # Берем веса для данной категории и всех тональностей
                    category_sentiment_weights_tensor = torch.tensor(
                        self.category_sentiment_weights[i], device=sentiment_logits.device
                    )
                    loss = F.cross_entropy(present_logits, target_indices, weight=category_sentiment_weights_tensor)
                else:
                    loss = F.cross_entropy(present_logits, target_indices)
                
                sentiment_losses.append(loss)
        
        sentiment_loss = torch.stack(sentiment_losses).mean() if sentiment_losses else torch.tensor(0.0, device=sentiment_logits.device)
        
        # Комбинированная функция потерь
        total_loss = (self.category_weight * category_loss + 
                     self.sentiment_weight * sentiment_loss)
        
        return total_loss, category_loss, sentiment_loss

def calculate_metrics(predictions, targets, task_type='category'):
    """
    Вычисляет метрики для оценки модели
    
    Args:
        predictions: Предсказания модели
        targets: Целевые значения
        task_type: Тип задачи ('category' или 'sentiment')
    
    Returns:
        metrics: Словарь с метриками
    """
    if task_type == 'category':
        # Для мультилейбл классификации
        preds = torch.sigmoid(predictions) > 0.5
        
        # Точность (accuracy)
        accuracy = (preds == targets).float().mean()
        
        # Micro-averaged F1 (глобальные TP, FP, FN)
        tp_micro = (preds * targets).sum().float()
        fp_micro = (preds * (~targets.bool())).sum().float()
        fn_micro = ((~preds.bool()) * targets).sum().float()
        
        micro_precision = tp_micro / (tp_micro + fp_micro + 1e-8)
        micro_recall = tp_micro / (tp_micro + fn_micro + 1e-8)
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-8)
        
        # Macro-averaged F1 (среднее по классам)
        tp = (preds * targets).sum(dim=0)
        fp = (preds * (~targets.bool())).sum(dim=0)
        fn = ((~preds.bool()) * targets).sum(dim=0)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        f1_macro = f1.mean()
        
        return {
            'accuracy': accuracy.item(),
            'micro_f1': micro_f1.item(),
            'micro_precision': micro_precision.item(),
            'micro_recall': micro_recall.item(),
            'macro_f1': f1_macro.item(),
            'macro_precision': precision.mean().item(),
            'macro_recall': recall.mean().item()
        }
    
    else:  # sentiment
        # Для классификации тональности каждой категории
        # predictions: (batch_size, num_categories, 4)
        # targets: (batch_size, num_categories, 4)
        
        # Получаем предсказанные классы для каждой категории
        pred_classes = torch.argmax(predictions, dim=-1)  # (batch_size, num_categories)
        target_classes = torch.argmax(targets, dim=-1)    # (batch_size, num_categories)
        
        # Вычисляем точность для каждой категории
        accuracy_per_category = (pred_classes == target_classes).float().mean(dim=0)
        overall_accuracy = accuracy_per_category.mean()
        
        return {
            'accuracy': overall_accuracy.item(),
            'accuracy_per_category': accuracy_per_category.cpu().numpy()
        }
