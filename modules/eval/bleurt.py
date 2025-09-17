# evaluators/bleurt.py

from modules.eval.base_evaluator import BaseEvaluator
from modules.eval.registry import EvaluatorRegistry
import torch
import evaluate
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
@EvaluatorRegistry.register("bleurt")
class BLEURTEvaluator(BaseEvaluator):
    """
    Evaluator that calculates the accuracy of the model.
    """
    def __init__(self, target_instance, args, model_path="/home/model/BLEURT-20"):
        """
        Initializes the BLEURT model and tokenizer.
        
        Parameters:
        - model_path (str): Path to the BLEURT model directory. Defaults to "/home/model/BLEURT-20-D12".
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        
    def get_name(self):
        return 'bleurt'
    
    def load_model(self):
        """Load the BLEURT model and tokenizer."""
        try:
            print(f"Loading BLEURT model from {self.model_path}")
            config = BleurtConfig.from_pretrained(self.model_path)
            self.model = BleurtForSequenceClassification.from_pretrained(self.model_path, config=config)
            self.tokenizer = BleurtTokenizer.from_pretrained(self.model_path)
            self.model = self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode
            print("Model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def evaluate(self, dataset,item_name,reference_key,candidate_key):
        data_items = dataset.get_data()
        print(data_items)
        for item in data_items:
            reference = item[reference_key]
            candidate = item[candidate_key]
            item[item_name] = self.meta_evaluate(item, reference,candidate)

    def meta_evaluate(self, item, reference, candidate):
        """
        Calculate accuracy.

        Args:
            model: The model to evaluate. Assumes it has a `predict` method.
            data: A tuple (inputs, labels).

        Returns:
            A dictionary with accuracy.
        """
        with torch.no_grad():
            inputs = self.tokenizer(reference,candidate,padding='max_length', truncation=True, max_length=512, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return self.model(**inputs).logits.flatten().tolist()[0]
        
    def get_score(self, context, response, ground_truth, context2=None):
        item = []
        return round(self.meta_evaluate(item, ground_truth, response) * 100, 1)