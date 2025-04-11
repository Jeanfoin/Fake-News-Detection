import torch
from transformers import DistilBertModel, DistilBertConfig

class DistilBERTWDropout(torch.nn.Module):
    """
    A DistilBERT model with dropout for text classification.

    Parameters:
    -----------
    distilbert_model_name (str, optional): The name of the pretrained DistilBERT model to use. Default is "distilbert-base-cased".
    dropout_rate (float, optional): The dropout rate to apply. Default is 0.1.
    max_tokens (int, optional): The maximum number of tokens for the model. Default is 512.

    Attributes:
    -----------
    distilbert_model_name (str): The name of the pretrained DistilBERT model.
    dropout (torch.nn.Dropout): The dropout layer.
    max_tokens (int): The maximum number of tokens for the model.
    distilbert (DistilBertModel): The DistilBERT model.
    classifier (torch.nn.Linear): The classification layer.
    """
    def __init__(
        self,
        
        distilbert_model_name: str = "distilbert-base-cased",
        dropout_rate: float = 0.1,
        max_tokens: int=512
    ):
        super().__init__()

        self.distilbert_model_name = distilbert_model_name
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.max_tokens = max_tokens
        
        config = DistilBertConfig.from_pretrained(self.distilbert_model_name, max_position_embeddings=self.max_tokens)
        self.distilbert = DistilBertModel.from_pretrained(self.distilbert_model_name, 
                                                          config=config,
                                                          ignore_mismatched_sizes=True)
        self.classifier = torch.nn.Linear(self.distilbert.config.hidden_size, 1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Parameters:
        -----------
        input_ids : torch.Tensor
            Tensor of token IDs, shaped (batch_size, sequence_length).
        attention_mask : torch.Tensor
            Tensor indicating the positions of valid tokens, shaped (batch_size, sequence_length).
            
        Returns:
        --------
        logits : torch.Tensor
            The predicted logits for each label, shaped (batch_size, num_labels).
        """
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)        
        pooled_output = outputs.last_hidden_state[:, 0, :]  
        pooled_output = self.dropout(pooled_output)  
        logits = self.classifier(pooled_output)
        return logits
