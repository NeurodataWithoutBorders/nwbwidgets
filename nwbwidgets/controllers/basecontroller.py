from abc import abstractmethod, ABC
from typing import Dict

import ipywidgets as widgets


class BaseController(ABC):
    """
    Base definition of all Controllers.

    A Controller is a container of objects such as widgets, including other controllers, that exposes all important
    components as non-private attributes at the outermost level for simplified reference.

    This is in constrast to defining an ipywidget.Box of other Boxes, where the only way to reference a particular
    sub-widget component is by navigating the children tree, knowing the set of levels and indices required to find
    a particular child.
    """

    def __init__(self, components: Dict[str, object]):
        """
        Initialize this controller given the pre-initialized set of components.

        Parameters
        ----------
        components: dictionary
            Used to map string names to widgets.
        """
        self.setup_components(components=components)
        self.setup_observers()

    @abstractmethod
    def setup_components(self, components: Dict[str, object]):
        """Define how to set the components given a dictionary of string mappings to arbitrary object types."""
        raise NotImplementedError("This Controller has not defined how to construct a Box container for its children!")

    def setup_observers(self):
        """
        Define observation events specific to the interactions and values of components within the same Controller.
        """
        pass  # Instead of NotImplemented; a given widget may not need or want to use any local observers

    def get_fields(self) -> Dict[str, object]:
        """
        Return the custom attributes set at the outer level for this Controller.

        Slightly more proper and better-looking than directly accessing the magic __dict__ attribute.

        Returns
        -------
        fields: dict
            The non-private attributes of this controller exposed at the outer-most level.
            These can be widgets, other controllers, or even mutable references.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and k != "components"}

    @abstractmethod
    def make_box(self, box_type: widgets.Box) -> widgets.Box:
        """
        Create a widget box container for the components of this widget.

        Parameters
        ----------
        box_type: ipywidgets.Box
            Any subtype of an ipywidgets Box base class.

        Returns
        -------
        boxed_controller: widget.Box
            The box of `box_type` populated with the components of this controller, which themselves may be contained
            in other boxes to obtain a desired layout.
        """
        raise NotImplementedError("This Controller has not defined how to construct a Box container for its children!")
