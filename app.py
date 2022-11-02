import lightning as L
import subprocess

class MLFlowServerWork(L.LightningWork):
    def __init__(self):
        super().__init__()
    
    def run(self):
        cmd = f"mlflow server --host 0.0.0.0 --port {self.port}"
        subprocess.run(cmd, shell=True)

class MLFlowApp(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.mlflow_server = MLFlowServerWork()
    
    def run(self, *args, **kwargs) -> None:
        self.mlflow_server.run()
        print(self.mlflow_server.url)
    
    def configure_layout(self):
        return {"name": None, "content": self.mlflow_server.url}

app = L.LightningApp(MLFlowApp())
