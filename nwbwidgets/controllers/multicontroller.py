"""
Class definition of the MultiController.

This class is used to efficiently combine any number of other Controllers by unpacking their contents and exposing
their atttributes at the outer level as opposed to having to perform nested calls to the ipywidgets.Box.children tuples.
"""
from typing import Dict, Union, Any

import ipywidgets

from .basecontroller import BaseController


class MultiController(BaseController):
    """Extension of the default Controller class specifically designed to unpack nested Controllers."""

    def _check_attribute_name_collision(self, name: str):
        """Enforce string name references to sub-widget or sub-controller objects to be unique."""
        if hasattr(self, name):
            raise KeyError(
                f"This Controller already has an outer attribute with the name '{name}'! "
                "Please adjust the reference key to be unique."
            )

    def _unpack_attributes(self, controller: BaseController):
        """If a component is a Controller, unpack its own attributes for outermost exposure."""
        for field_name, field in controller.get_fields().items():
            self._check_attribute_name_collision(name=field_name)
            setattr(self, field_name, field)

    def __init__(self, attributes: Dict[str, Union[BaseController, ipywidgets.Widget]]):
        super(BaseController, self).__init__()  # Setup Box properties
        self.layout.display = "flex"
        self.layout.align_items = "stretch"

        self.setup_attributes(attributes=attributes)
        self.setup_children()
        self.setup_observers()

    def setup_attributes(self, attributes: Dict[str, Union[BaseController, ipywidgets.Widget]]):
        components = dict()
        states = dict()
        self._propagate_setup_observers = list()
        for attribute_name, attribute in attributes.items():
            if isinstance(attribute, BaseController):
                self._unpack_attributes(controller=attribute)
                components.update({attribute_name: attribute})
                self._propagate_setup_observers.append(attribute.setup_observers)
            elif isinstance(attribute, ipywidgets.Widget):
                self._check_attribute_name_collision(name=attribute_name)
                setattr(self, attribute_name, attribute)
                components.update({attribute_name: attribute})
            else:
                self._check_attribute_name_collision(name=attribute_name)
                setattr(self, attribute_name, attribute)
                states.update({attribute_name: attribute})
        self.components = components  # Maintain sub-component structure for provenance
        self.states = states

    def setup_children(self):
        children = list()
        for component in self.components.values():
            if isinstance(component, BaseController):
                children.append(component)
            elif isinstance(component, ipywidgets.Widget):
                children.append(component)
        self.children = tuple(children)

    def setup_observers(self):
        for setup_observers in self._propagate_setup_observers:
            setup_observers()

    def get_fields(self) -> Dict[str, Any]:
        return {
            attribute_name: attribute
            for attribute_name, attribute in self.__dict__.items()
            if not attribute_name.startswith("_")  # Skip private attributes
            and attribute_name != "components"
            and attribute_name != "states"
        }
