from hdmf.testing import TestCase
from ipywidgets import Button, Checkbox

from nwbwidgets.controllers import GenericController


class ExampleGenericController(GenericController):
    def __init__(self, components: dict):
        self.custom_attribute = "abc"
        super().__init__(components=components)


class ExampleControllerWithObservers(GenericController):
    def __init__(self):
        super().__init__(components=dict(button=Button(), check_box=Checkbox()))

    def check_box_on_button_click(self, change):
        if self.check_box.value is True:
            self.check_box.value = False
        elif self.check_box.value is False:
            self.check_box.value = True

    def setup_observers(self):
        self.button.on_click(self.check_box_on_button_click)


class TestGenericController(TestCase):
    def test_generic_controller_component_controller_assertion(self):
        components = dict(other_controller=ExampleControllerWithObservers())

        with self.assertRaisesWith(
            exc_type=ValueError,
            exc_msg=(
                "Component 'other_controller' is a type of Controller! "
                "Please use a MultiController to unpack its components."
            ),
        ):
            ExampleGenericController(components=components)

    def test_generic_controller_name_collision(self):
        components = dict(button=Button(), check_box=Checkbox(), custom_attribute=123)

        with self.assertRaisesWith(
            exc_type=KeyError,
            exc_msg=(
                "\"This Controller already has an outer attribute with the name 'custom_attribute'! "
                'Please adjust the reference key to be unique."'
            ),
        ):
            ExampleGenericController(components=components)

    def test_generic_controller_no_components(self):
        controller = GenericController(components=dict())

        self.assertDictEqual(d1=controller.components, d2=dict())
        self.assertDictEqual(d1=controller.get_fields(), d2=dict())
        self.assertTupleEqual(tuple1=controller.children, tuple2=())

    def test_generic_controller_ipywidget_components(self):
        button = Button()
        check_box = Checkbox()
        components = dict(button=button, check_box=check_box)
        controller = GenericController(components=components)

        expected_components = dict(button=button, check_box=check_box)
        expected_fields = dict(button=button, check_box=check_box)
        expected_children = (button, check_box)
        self.assertDictEqual(d1=controller.components, d2=expected_components)
        self.assertDictEqual(d1=controller.get_fields(), d2=expected_fields)
        self.assertTupleEqual(tuple1=controller.children, tuple2=expected_children)

    def test_generic_controller_standard_attributes(self):
        """Non-widget attributes were not included in the children of the box."""
        button = Button()
        check_box = Checkbox()
        components = dict(
            button=button,
            check_box=check_box,
            some_integer=1,
            some_string="test",
            some_float=1.23,
            some_list=[1, 2, 3],
            some_dict=dict(a=5, b=6, c=7),
        )
        controller = GenericController(components=components)

        expected_components = dict(button=button, check_box=check_box)
        expected_fields = dict(
            button=button,
            check_box=check_box,
            some_integer=1,
            some_string="test",
            some_float=1.23,
            some_list=[1, 2, 3],
            some_dict=dict(a=5, b=6, c=7),
        )
        expected_children = (button, check_box)
        self.assertDictEqual(d1=controller.components, d2=expected_components)
        self.assertDictEqual(d1=controller.get_fields(), d2=expected_fields)
        assert controller.children == expected_children
        self.assertTupleEqual(tuple1=controller.children, tuple2=expected_children)

    def test_generic_controller_with_observers(self):
        controller = ExampleControllerWithObservers()

        assert controller.check_box.value is False

        controller.button.click()

        assert controller.check_box.value is True

        controller.button.click()

        assert controller.check_box.value is False
