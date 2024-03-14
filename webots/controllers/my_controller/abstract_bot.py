from abc import ABC, abstractmethod

class Robot(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def move(self):
        pass

    @abstractmethod
    def stop(self):
        pass

