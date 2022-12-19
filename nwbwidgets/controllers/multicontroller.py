from typing import Dict, Union

import ipywidgets

from .basecontroller import BaseController
from .genericcontroller import GenericController


class MultiController(GenericController):
    """Extension of the default Controller class specifically designed to unpack nested Controllers."""

    def _unpack_attributes(self, component):
        """If a component is a Controller, recurse its own attributes to move the level of outermost exposure."""
        for attribute_name, attribute in component.get_fields().items():
            self._check_attribute_name_collision(name=attribute_name)
            setattr(self, attribute_name, attribute)

    def setup_components(self, components: Dict[str, Union[ipywidgets.Widget]]):
        unpacked_components = dict()
        for component_name, component in components.items():
            if isinstance(component, BaseController):
                self._unpack_attributes(component=component)  # Unpack attributes to new outermost level
                unpacked_components.update({component_name: component.components})  # Nested dictionary
            else:
                self._check_attribute_name_collision(name=component_name)
                setattr(self, component_name, component)
                unpacked_components.update({component_name: component})
        self.components = unpacked_components  # Maintain sub-component structure for provenance

    def make_box(self, box_type: ipywidgets.Box) -> ipywidgets.Box:
        children = list()
        for child in self.get_fields().values():
            if isinstance(child, ipywidgets.Widget):
                children.append(child)
            elif isinstance(child, BaseController):
                children.append(child.make_box(box_type=box_type))
        return box_type(children=tuple(children))
