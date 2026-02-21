"""
Training and validation functions
"""
import torch
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, config, qat_enabled=False):
    """Train for one epoch with mixed precision and gradient accumulation
    
    Args:
        qat_enabled: If True, disables mixed precision (QAT requires FP32)
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    logger.info(f"Starting Epoch {epoch}")
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
    
    # Disable mixed precision when QAT is active
    use_mixed_precision = config.USE_MIXED_PRECISION and scaler is not None and not qat_enabled
    
    for batch_idx, (images, targets) in enumerate(pbar):
        try:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Mixed precision forward pass (only if not using QAT)
            if use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    losses = losses / config.ACCUMULATION_STEPS  # Scale for gradient accumulation
                
                # Mixed precision backward
                scaler.scale(losses).backward()
                
                # Update weights every ACCUMULATION_STEPS
                if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
                    if config.GRADIENT_CLIP_VALUE > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # FP32 training (required for QAT)
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses = losses / config.ACCUMULATION_STEPS
                losses.backward()
                
                if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
                    if config.GRADIENT_CLIP_VALUE > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += losses.item() * config.ACCUMULATION_STEPS
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {losses.item():.4f}")
            
            pbar.set_postfix({
                'loss': f'{losses.item():.4f}',
                'avg': f'{total_loss / (batch_idx + 1):.4f}'
            })
            
        except Exception as e:
            logger.error(f"Error in Epoch {epoch}, Batch {batch_idx}: {str(e)}", exc_info=True)
            raise
    
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {epoch} completed. Average training loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, dataloader, device):
    """Validate the model"""
    model.train()  # Keep in train mode for loss calculation
    total_loss = 0
    
    logger.info("Starting validation...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Validating", leave=False)):
            try:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                
            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {str(e)}", exc_info=True)
                raise
    
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Validation completed. Average validation loss: {avg_loss:.4f}")
    return avg_loss
