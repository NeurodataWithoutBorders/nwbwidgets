"""
Base class definition of all controllers.

Attempted to make the class abstract via `abc.ABC` to use `abc.abstractmethod` but ran into metaclass conflict
issues with `ipywidgets.Box`. Undefined methods instead raise NotImplementedErrors.
"""
from typing import Dict, Any

import ipywidgets


class BaseController(ipywidgets.Box):
    """
    Base definition of all Controllers.

    A Controller is a container of objects such as widgets, including other controllers, that exposes all important
    components as non-private attributes at the outermost level for simplified reference.

    This is in constrast to defining an ipywidget.Box of other Boxes, where the only way to reference a particular
    sub-widget component is by navigating the children tree, knowing the set of levels and indices required to find
    a particular child.
    """

    def __init__(self):
        """
        Defines the workflow for initializing a Controller.

        Overriding this method in a child class without calling super().__init__() is highly discouraged.

        To align the children vertically, set `your_controller.layout.flex_flow = "column"`.
        """
        super().__init__()  # Setup Box properties
        self.layout.display = "flex"
        self.layout.align_items = "stretch"

        self.setup_attributes()
        self.setup_children()
        self.setup_observers()

    def setup_attributes(self):
        """
        Define how to setup the widget components and non-widget state tracking values for this Controller.

        These must be defined as non-private attributes exposed at the outer level.

        E.g.,

        >>> self.run_button = ipywidgets.Button(description="Run...")
        >>> self.count_button_presses: int = 0
        """
        raise NotImplementedError("This Controller has not defined how to setup its attributes!")

    def setup_children(self):
        """Define how to layout the children of this box."""
        raise NotImplementedError("This Controller has not defined how to layout its children!")

    def setup_observers(self):
        """
        Define observation events specific to the interactions between components and statess within this Controller.
        """
        # Instead of raising NotImplementedError or being an abstractmethod,
        # a given widget may not need or want to use any local observers.
        pass

    def get_fields(self) -> Dict[str, Any]:
        """
        Return the all attributes set at the outer level for this Controller.

        Slightly more proper and better-looking than directly accessing the magic __dict__ attribute.

        Returns
        -------
        fields : dict
            The non-private attributes of this controller exposed at the outer-most level.
            These can be widgets, other controllers, or even mutable references.
        """
        return {
            attribute_name: attribute
            for attribute_name, attribute in self.__dict__.items()
            if not attribute_name.startswith("_")
        }
