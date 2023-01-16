from hdmf.testing import TestCase
from ipywidgets import Button, Checkbox, Dropdown, ToggleButton

from nwbwidgets.controllers import BaseController, MultiController


class ExampleMultiController(MultiController):
    def __init__(self, attributes: dict):
        self.custom_attribute = "abc"
        super().__init__(attributes=attributes)


class ExampleControllerWithObservers(BaseController):
    def setup_attributes(self):
        self.button = Button()
        self.check_box = Checkbox()

    def setup_children(self):
        self.children = (self.button, self.check_box)

    def check_box_on_button_click(self, change):
        if self.check_box.value is True:
            self.check_box.value = False
        elif self.check_box.value is False:
            self.check_box.value = True

    def setup_observers(self):
        self.button.on_click(self.check_box_on_button_click)


class ExampleMultiControllerWithObservers(MultiController):
    def __init__(self):
        super().__init__(
            components=dict(
                example_controller_1=ExampleMultiController(components=dict()),
                example_controller_2=ExampleControllerWithObservers(),
            )
        )

    def adjust_custom_attribute_on_button_click(self, change):
        if self.custom_attribute == "abc":
            self.custom_attribute = "def"
        elif self.custom_attribute == "def":
            self.custom_attribute = "abc"

    def setup_observers(self):
        self.button.on_click(self.adjust_custom_attribute_on_button_click)


class TestMultiController(TestCase):
    def test_multi_controller_name_collision_basic(self):
        components = dict(button=Button(), check_box=Checkbox(), custom_attribute=123)

        with self.assertRaisesWith(
            exc_type=KeyError,
            exc_msg=(
                "\"This Controller already has an outer attribute with the name 'custom_attribute'! "
                'Please adjust the reference key to be unique."'
            ),
        ):
            ExampleMultiController(components=components)

    # TODO: fix tests to rely on base rather than generic
    # def test_name_collision_nested(self):
    #     components = dict(
    #         button=Button(),
    #         check_box=Checkbox(),
    #         other_controller=GenericController(components=dict(button=Button(), toggle_button=ToggleButton())),
    #     )

    #     with self.assertRaisesWith(
    #         exc_type=KeyError,
    #         exc_msg=(
    #             "\"This Controller already has an outer attribute with the name 'button'! "
    #             'Please adjust the reference key to be unique."'
    #         ),
    #     ):
    #         MultiController(components=components)

    def test_multi_controller_no_components(self):
        multi_controller = MultiController(components=dict())

        self.assertDictEqual(d1=multi_controller.components, d2=dict())
        self.assertDictEqual(d1=multi_controller.get_fields(), d2=dict())
        self.assertTupleEqual(tuple1=multi_controller.children, tuple2=())

    # def test_multi_controller_ipywidget_and_controller_components(self):
    #     button = Button()
    #     dropdown = Dropdown()
    #     check_box = Checkbox()
    #     other_controller = GenericController(components=dict(dropdown=dropdown, check_box=check_box))
    #     multi_controller = MultiController(components=dict(button=button, other_controller=other_controller))

    #     expected_components = dict(button=button, other_controller=other_controller)
    #     expected_fields = dict(button=button, dropdown=dropdown, check_box=check_box)
    #     expected_children = (button, other_controller)
    #     self.assertDictEqual(d1=multi_controller.components, d2=expected_components)
    #     self.assertDictEqual(d1=multi_controller.get_fields(), d2=expected_fields)
    #     self.assertTupleEqual(tuple1=multi_controller.children, tuple2=expected_children)

    # def test_multi_controller_standard_attributes(self):
    #     """Non-widget attributes were not included in the children of the box."""
    #     button = Button()
    #     check_box = Checkbox()
    #     some_integer = (1,)
    #     some_float = (1.23,)
    #     some_dict = (dict(a=5, b=6, c=7),)
    #     some_string = ("test",)
    #     some_list = ([1, 2, 3],)

    #     components1 = dict(button=button, some_integer=some_integer, some_float=some_float, some_dict=some_dict)
    #     controller1 = GenericController(components=components1)
    #     components2 = dict(check_box=check_box, some_string=some_string, some_list=some_list)
    #     controller2 = GenericController(components=components2)

    #     multi_controller = MultiController(components=dict(controller1=controller1, controller2=controller2))

    #     expected_components = dict(controller1=controller1, controller2=controller2)
    #     expected_fields = dict(
    #         button=button,
    #         some_integer=some_integer,
    #         some_float=some_float,
    #         some_dict=some_dict,
    #         check_box=check_box,
    #         some_string=some_string,
    #         some_list=some_list,
    #     )
    #     expected_children = (controller1, controller2)
    #     self.assertDictEqual(d1=multi_controller.components, d2=expected_components)
    #     self.assertDictEqual(d1=multi_controller.get_fields(), d2=expected_fields)
    #     self.assertTupleEqual(tuple1=multi_controller.children, tuple2=expected_children)

    def test_multi_controller_with_observers(self):
        multi_controller = ExampleMultiControllerWithObservers()

        assert multi_controller.check_box.value is False
        assert multi_controller.custom_attribute == "abc"

        multi_controller.button.click()

        assert multi_controller.check_box.value is True
        assert multi_controller.custom_attribute == "def"

        multi_controller.button.click()

        assert multi_controller.check_box.value is False
        assert multi_controller.custom_attribute == "abc"
