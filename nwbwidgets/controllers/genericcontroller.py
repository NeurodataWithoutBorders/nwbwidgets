from typing import Dict

import ipywidgets

from .basecontroller import BaseController


class GenericController(BaseController):
    """Default all-purpose controller class."""

    def _check_attribute_name_collision(self, name: str):
        """Enforce string name references to sub-widget or sub-controller objects to be unique."""
        if hasattr(self, name):
            raise KeyError(
                f"This Controller already has an outer attribute with the name '{name}'! "
                "Please adjust the reference key to be unique."
            )

    def setup_components(self, components: Dict[str, object]):
        unpacked_components = dict()
        for component_name, component in components.items():
            if isinstance(component, BaseController):
                raise ValueError(
                    f"Component '{component_name}' is a type of Controller! "
                    "Please use a MultiController to unpack its components."
                )
            elif isinstance(component, ipywidgets.Widget):
                self._check_attribute_name_collision(name=component_name)
                setattr(self, component_name, component)
                unpacked_components.update({component_name: component})
            else:
                self._check_attribute_name_collision(name=component_name)
                setattr(self, component_name, component)
        self.components = unpacked_components  # Maintain sub-component structure for provenance

    def setup_children(self):
        # A simple default rule for generating a box layout for this Controllers components.
        # It is usually recommended to override with a custom layout for a specific visualization.
        self.children = tuple(child for child in self.components.values())
