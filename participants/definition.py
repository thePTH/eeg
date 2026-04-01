from participants.groups import Group
from participants.genders import Gender
from typing import Union
from utils.enum import EnumParser




class Participant :
    def __init__(self, id:str, gender:Union[Gender, str], age:int, group:Union[Group,str], mmse:int):

        
        self._gender = EnumParser.parse(gender, Gender)
        self._group = EnumParser.parse(group, Group)

        self._id = id
        self._age = age
        self._mmse = mmse


    @property
    def id(self):
        return self._id
    @property
    def gender(self) ->str :
        return self._gender.value
    @property
    def age(self):
        return self._age
    @property
    def group(self) -> str:
        return self._group.value
    @property
    def health_state(self) -> str:
        return self._group.health_state.value
    @property
    def mmse(self):
        return self._mmse
    
    def to_dict(self):
        return {"id": self.id, "gender":self.gender, "age":self.age, "group":self.group, "mmse":self.mmse}

    

class ParticipantFactory:
    @staticmethod
    def build(dico:dict):
        return Participant(id=dico["id"], gender=dico["gender"], age=dico["age"], group=dico["group"], mmse=dico["mmse"])
    