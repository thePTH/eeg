from enum import Enum

class HealthState(Enum):
    AD = "Alzheimer"
    FTD = "Frontotemporal Dementia"
    CN = "Healthy"

class Group(Enum):
    A = "AD"
    F = "FTD"
    C = "CN"
    
    @property
    def health_state(self):
        return HealthState[self.value]