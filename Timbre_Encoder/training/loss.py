"""
Contrastive Loss for Speaker Verification
==========================================

InfoNCE (Noise Contrastive Estimation) loss for learning speaker embeddings.

Given a batch with N speakers × K utterances:
    - Anchor: utt_A1
    - Positive: utt_A2 (same speaker, different content)
    - Negatives: All other utterances (different speakers)

Loss = -log[ exp(sim(anchor, positive) / τ) /
              Σ_n exp(sim(anchor, n) / τ) ]

Temperature τ: Controls hardness of negatives (typical: 0.05)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = get_logger(__name__)


class InfoNCELoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.05,
        use_hard_negatives: bool = False,
        hard_negative_threshold: float = 0.5,
        hard_negative_weight: float = 2.0,
    ):
        super().__init__()

        logger.info("Initializing InfoNCELoss")
        logger.info(f"Parameters: temperature={temperature}, use_hard_negatives={use_hard_negatives}")
        if use_hard_negatives:
            logger.info(f"Hard negative mining: threshold={hard_negative_threshold}, weight={hard_negative_weight}")

        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_threshold = hard_negative_threshold
        self.hard_negative_weight = hard_negative_weight
    
    def forward(
        self,
        embeddings: torch.Tensor,
        speakers_per_batch: int,
        utterances_per_speaker: int,
    ) -> dict:
        """
        Compute InfoNCE loss (VECTORIZED - no Python loops).
        """
        batch_size = embeddings.size(0)
        assert batch_size == speakers_per_batch * utterances_per_speaker

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix [B, B]
        similarity = torch.matmul(embeddings, embeddings.t())
        
        # Create masks (vectorized)
        positive_mask = self._create_positive_mask(
            speakers_per_batch,
            utterances_per_speaker,
            device=embeddings.device
        )
        negative_mask = 1 - positive_mask - torch.eye(batch_size, device=embeddings.device)
        
        # Scale by temperature
        similarity = similarity / self.temperature
        
        # Compute positive similarities: average over positives for each anchor
        # positive_mask has K-1 positives per row
        pos_count = positive_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pos_sim = (similarity * positive_mask).sum(dim=1) / pos_count.squeeze()  # [B]
        
        # Compute negative log-sum-exp
        # Set non-negatives to -inf so they don't contribute to logsumexp
        neg_similarity = similarity.clone()
        neg_similarity[negative_mask == 0] = float('-inf')
        neg_logsumexp = torch.logsumexp(neg_similarity, dim=1)  # [B]
        
        # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = -pos + log(exp(pos) + sum(exp(neg)))
        # = -pos + log(exp(pos) + exp(neg_logsumexp))
        # = -pos + logsumexp([pos, neg_logsumexp])
        loss_per_sample = -pos_sim + torch.logsumexp(
            torch.stack([pos_sim, neg_logsumexp], dim=1), dim=1
        )
        loss = loss_per_sample.mean()
        
        # Compute accuracy: is max positive > max negative?
        # NOTE: was previously (similarity * mask + (1-mask)*-inf).max() which
        # produces 0*-inf = NaN where mask=1, poisoning the max to NaN → accuracy
        # always 0. Use masked_fill on a clone to avoid that.
        sim_pos = similarity.masked_fill(positive_mask == 0, float('-inf'))
        sim_neg = similarity.masked_fill(negative_mask == 0, float('-inf'))
        max_pos = sim_pos.max(dim=1)[0]
        max_neg = sim_neg.max(dim=1)[0]
        accuracy = (max_pos > max_neg).float().mean().item()
        
        # Hard negative ratio (simplified - count negatives above threshold)
        hard_negative_ratio = 0.0
        if self.use_hard_negatives:
            hard_mask = (similarity * self.temperature > self.hard_negative_threshold) & (negative_mask > 0)
            hard_negative_ratio = hard_mask.float().sum().item() / negative_mask.sum().item()

        return {
            'loss': loss,
            'accuracy': accuracy,
            'hard_negative_ratio': hard_negative_ratio,
        }
    
    def _create_positive_mask(
        self,
        speakers_per_batch: int,
        utterances_per_speaker: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create positive pair mask (VECTORIZED - no Python loops).
        
        For batch structure: [spk0_utt0, spk0_utt1, spk1_utt0, spk1_utt1, ...]
        
        Returns:
            mask: [B, B] where mask[i, j] = 1 if i, j are same speaker, different utterance
        """
        batch_size = speakers_per_batch * utterances_per_speaker
        
        # Create speaker labels for each sample: [0,0,0,1,1,1,2,2,2,...] for K=3
        speaker_labels = torch.arange(speakers_per_batch, device=device).repeat_interleave(utterances_per_speaker)
        
        # Same speaker mask: speaker_labels[i] == speaker_labels[j]
        mask = (speaker_labels.unsqueeze(0) == speaker_labels.unsqueeze(1)).float()
        
        # Remove diagonal (same utterance)
        mask = mask - torch.eye(batch_size, device=device)
        
        return mask
    
    def _compute_hard_negative_weights(
        self,
        negatives: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weights for hard negative mining.
        
        Hard negatives: Different speaker but high similarity (confusing pairs)
        """
        # Start with uniform weights
        weights = negative_mask.clone()
        
        # Find hard negatives (similarity > threshold)
        hard_mask = (negatives > self.hard_negative_threshold) & (negative_mask > 0)
        
        # Increase weight for hard negatives
        weights[hard_mask] = weights[hard_mask] * self.hard_negative_weight
        
        return weights


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (alternative to InfoNCE).
    
    Similar to InfoNCE but pulls all positives closer while pushing negatives away.
    """
    
    def __init__(
        self,
        temperature: float = 0.05,
    ):
        super().__init__()
        logger.info(f"Initializing SupConLoss with temperature={temperature}")
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        speakers_per_batch: int,
        utterances_per_speaker: int,
    ) -> dict:
        """Compute supervised contrastive loss."""
        batch_size = embeddings.size(0)
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Similarity matrix
        similarity = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Create masks
        positive_mask = self._create_positive_mask(
            speakers_per_batch, utterances_per_speaker, embeddings.device
        )
        negative_mask = 1 - positive_mask - torch.eye(batch_size, device=embeddings.device)
        
        # For each anchor
        losses = []
        for i in range(batch_size):
            # Positives for this anchor
            pos_similarities = similarity[i] * positive_mask[i]
            num_positives = positive_mask[i].sum()
            
            if num_positives == 0:
                continue
            
            # Compute log sum exp over positives
            log_sum_exp_pos = torch.logsumexp(pos_similarities[positive_mask[i] > 0], dim=0)
            
            # Compute log sum exp over all (positives + negatives)
            mask_all = positive_mask[i] + negative_mask[i]
            log_sum_exp_all = torch.logsumexp(similarity[i][mask_all > 0], dim=0)
            
            # Loss for this anchor
            loss = -(log_sum_exp_pos - log_sum_exp_all) / num_positives
            losses.append(loss)
        
        loss = torch.stack(losses).mean()
        
        return {
            'loss': loss,
            'accuracy': 0.0,  # Not computed for SupCon
            'hard_negative_ratio': 0.0,
        }
    
    def _create_positive_mask(self, speakers_per_batch, utterances_per_speaker, device):
        """Same as InfoNCELoss (VECTORIZED)."""
        batch_size = speakers_per_batch * utterances_per_speaker
        
        # Create speaker labels for each sample
        speaker_labels = torch.arange(speakers_per_batch, device=device).repeat_interleave(utterances_per_speaker)
        
        # Same speaker mask
        mask = (speaker_labels.unsqueeze(0) == speaker_labels.unsqueeze(1)).float()
        
        # Remove diagonal
        mask = mask - torch.eye(batch_size, device=device)
        
        return mask


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss (additive angular margin).
    
    Used for speaker identification (not verification).
    Learns discriminative embeddings by adding angular margin.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_speakers: int,
        margin: float = 0.5,
        scale: float = 30.0,
    ):
        super().__init__()

        logger.info("Initializing ArcFaceLoss")
        logger.info(f"Parameters: embedding_dim={embedding_dim}, num_speakers={num_speakers}, margin={margin}, scale={scale}")

        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.margin = margin
        self.scale = scale

        # Weight matrix [num_speakers, embedding_dim]
        self.weight = nn.Parameter(torch.randn(num_speakers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        logger.debug(f"Initialized weight matrix with shape {self.weight.shape}")
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """
        Compute ArcFace loss.
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)  # [B, num_speakers]
        
        # Get angle
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # Add margin to target class
        one_hot = F.one_hot(labels, self.num_speakers).float()
        theta_m = theta + one_hot * self.margin
        
        # Convert back to cosine
        cosine_m = torch.cos(theta_m)
        
        # Scale
        logits = cosine_m * self.scale
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Accuracy
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy.item(),
        }


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning.
    
    Combines contrastive loss with classification loss.
    """
    
    def __init__(
        self,
        contrastive_loss: nn.Module,
        classification_loss: nn.Module,
        contrastive_weight: float = 1.0,
        classification_weight: float = 0.1,
    ):
        super().__init__()

        logger.info("Initializing CombinedLoss")
        logger.info(f"Weights - Contrastive: {contrastive_weight}, Classification: {classification_weight}")

        self.contrastive_loss = contrastive_loss
        self.classification_loss = classification_loss
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        speakers_per_batch: int,
        utterances_per_speaker: int,
    ) -> dict:
        """Compute combined loss."""
        # Contrastive loss
        contrastive_output = self.contrastive_loss(
            embeddings, speakers_per_batch, utterances_per_speaker
        )
        
        # Classification loss
        classification_output = self.classification_loss(embeddings, labels)
        
        # Combined loss
        loss = (
            self.contrastive_weight * contrastive_output['loss'] +
            self.classification_weight * classification_output['loss']
        )
        
        return {
            'loss': loss,
            'contrastive_loss': contrastive_output['loss'].item(),
            'classification_loss': classification_output['loss'].item(),
            'accuracy': contrastive_output['accuracy'],
        }