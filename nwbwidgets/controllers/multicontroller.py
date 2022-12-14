from typing import Tuple, Dict

import ipywidgets as widgets


class MultiController(widgets.VBox):
    controller_fields: Tuple[str] = tuple()
    components: Dict[str, widgets.VBox] = dict()

    def __init__(self, components: list):
        super().__init__()

        children = list()
        controller_fields = list()
        self.components = {component.__class__.__name__: component for component in components}
        for component in self.components.values():
            # Set attributes at outermost level
            for field in component.controller_fields:
                controller_fields.append(field)
                setattr(self, field, getattr(component, field))

            # Default layout of children
            if isinstance(component, widgets.Widget) and not isinstance(component, MultiController):
                children.append(component)

        self.children = tuple(children)
        self.controller_fields = tuple(controller_fields)

        self.setup_observers()

    def setup_observers(self):
        pass
