from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import logging

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")
        self.state: Dict[str, Any] = {}

    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        pass

    async def handle_error(self, error: Exception) -> Dict[str, Any]:
        self.logger.error(f"Error in {self.name}: {str(error)}")
        return {"status": "error", "error": str(error)}