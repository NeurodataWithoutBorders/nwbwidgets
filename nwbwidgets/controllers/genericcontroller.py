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
                "Please adjust the reference string to be unique."
            )

    def setup_components(self, components: Dict[str, object]):
        unpacked_components = dict()
        for component_name, component in components.items():
            if isinstance(component, BaseController):
                raise ValueError(
                    "Component '{component_name}' is a type of Controller - "
                    "use the MultiController to unpack its components!"
                )
            else:
                self._check_attribute_name_collision(name=component_name)
                setattr(self, component_name, component)
                unpacked_components.update({component_name: component})
        self.components = unpacked_components  # Maintain sub-component structure for provenance

    def get_fields(self) -> Dict[str, object]:
        """
        Slightly more proper or better-looking than directly accessing the magic __dict__ attribute.

        Returns
        -------
        fields: dict
            The non-private attributes of this controller exposed at the outer-most level.
            These can be widgets, other controllers, or even mutable references.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and k != "components"}

    def make_box(self, box_type: ipywidgets.Box) -> ipywidgets.Box:
        """
        A simple default rule for generating a box layout for this Controllers components.

        It is usually recommended to override with a custom layout for a specific visualization.
        """
        children = tuple(child for child in self.get_fields().values() if isinstance(child, ipywidgets.Widget))
        return box_type(children=children)
