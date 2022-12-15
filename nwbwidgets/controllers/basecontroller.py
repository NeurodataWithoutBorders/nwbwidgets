from abc import abstractmethod, ABC

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

    def _check_attribute_name_collision(self, name: str):
        """Enforce string name references to sub-widget or sub-controller objects to be unique."""
        if hasattr(self, name):
            raise KeyError(
                f"This Controller already has an outer attribute with the name {name}! "
                "Please adjust the reference string to be unique."
            )

    def _unpack_attributes(self, component):
        """If a component is a Controller, recurse its own attributes to move the level of outermost exposure."""
        for attribute_name, attribute in component.get_fields().items():
            self._check_attribute_name_collision(name=attribute_name)
            setattr(self, attribute_name, attribute)

    def __init__(self, components: dict):
        """
        Initialize this controller given the pre-initialized set of components.

        Parameters
        ----------
        components: dictionary
            Used to map string names to objects.
            The values of this dictionary can be either widgets or other controllers.
        """
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

        self.setup_observers()

    def setup_observers(self):
        """
        Define observation events specific to the interactions and values of components within the same MultiController.
        """
        pass  # Instead of NotImplemented; a given widget may not need or want to use any local observers

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

    def get_fields(self) -> dict:
        """
        Slightly more proper or better-looking than directly accessing the magic __dict__ attribute.

        Returns
        -------
        fields: dict
            The non-private attributes of this controller exposed at the outer-most level.
            These can be widgets, other controllers, or even mutable references.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and k != "components"}
