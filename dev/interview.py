from abc import ABC, abstractmethod
class Animal(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
    
    @abstractmethod
    def speak(self) -> None:
        pass

class Dog(Animal):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def speak(self) -> None:
        return "Dog"
    
class Cat(Animal):
    def __init__(self, name: str) -> None:
        super().__init__(name)
    
    def speak(self) -> None:
        return "Cat"
    

def talk_animal(animal: Animal) -> str:
    return animal.speak()

