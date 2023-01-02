from typing import Dict, Union

import ipywidgets

from .basecontroller import BaseController
from .genericcontroller import GenericController


class MultiController(GenericController):
    """Extension of the default Controller class specifically designed to unpack nested Controllers."""

    def _unpack_attributes(self, component):
        """If a component is a Controller, unpack its own attributes for outermost exposure."""
        for field_name, field in component.get_fields().items():
            self._check_attribute_name_collision(name=field_name)
            setattr(self, field_name, field)

    def setup_components(self, components: Dict[str, Union[ipywidgets.Widget]]):
        unpacked_components = dict()
        self._propagate_setup_observers = list()
        for component_name, component in components.items():
            if isinstance(component, BaseController):
                self._unpack_attributes(component=component)  # Unpack attributes to new outermost level
                unpacked_components.update({component_name: component.components})  # Nested dictionary
                self._propagate_setup_observers.append(component.setup_observers)
            else:
                self._check_attribute_name_collision(name=component_name)
                setattr(self, component_name, component)
                unpacked_components.update({component_name: component})
        self.components = unpacked_components  # Maintain sub-component structure for provenance

    def setup_children(self):
        children = list()
        for child in self.get_fields().values():
            if isinstance(child, ipywidgets.Widget):
                children.append(child)
            elif isinstance(child, BaseController):
                children.append(child.children)
        self.children = children

    def setup_observers(self):
        for setup_observers in self._propagate_setup_observers:
            setup_observers()
