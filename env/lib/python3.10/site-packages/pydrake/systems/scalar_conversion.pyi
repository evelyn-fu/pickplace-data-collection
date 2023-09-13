from _typeshed import Incomplete
from pydrake.common import pretty_class_name as pretty_class_name
from pydrake.common.cpp_template import TemplateClass as TemplateClass
from pydrake.systems.framework import LeafSystem_ as LeafSystem_, SystemScalarConverter as SystemScalarConverter

class TemplateSystem(TemplateClass):
    def __init__(self, name, T_list: Incomplete | None = ..., T_pairs: Incomplete | None = ..., scope: Incomplete | None = ...) -> None: ...
    @classmethod
    def define(cls, name, T_list: Incomplete | None = ..., T_pairs: Incomplete | None = ..., *args, scope: Incomplete | None = ..., **kwargs): ...
