# Temporary mock model registry for judging round

class ModelRegistry:
    def __init__(self):
        self.models = {}

    def get_model(self, name):
        return self.models.get(name)

    async def load_all(self):
        """
        Mock model loading for startup.
        Replace with actual ML model loading later.
        """
        print("✅ ModelRegistry: Mock models loaded.")

model_registry = ModelRegistry()