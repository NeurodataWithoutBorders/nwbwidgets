from hdmf.testing import TestCase
from ipywidgets import Button, Checkbox, HBox

from nwbwidgets.controllers import GenericController


class ExampleGenericController(GenericController):
    def __init__(self, components: dict):
        self.custom_attribute = "abc"
        super().__init__(components=components)


class ExampleGenericControllerWithObservers(GenericController):
    def __init__(self):
        super().__init__(components=dict(button=Button(), check_box=Checkbox()))

    def check_box_on_button_click(self, change):
        if self.check_box.value is True:
            self.check_box.value = False
        elif self.check_box.value is False:  # ipywidgets.CheckBox().value can be overridden to non-bool values
            self.check_box.value = True

    def setup_obervers(self):
        self.button.on_click(lambda change: self.check_box_on_button_click(change.new))


class TestGenericController(TestCase):
    def test_generic_controller_name_collision(self):
        button = Button()
        check_box = Checkbox()
        components = dict(button=button, check_box=check_box, custom_attribute=123)

        with self.assertRaisesWith(
            exc_type=KeyError,
            exc_msg=(
                "\"This Controller already has an outer attribute with the name 'custom_attribute'! "
                'Please adjust the reference string to be unique."'
            ),
        ):
            ExampleGenericController(components=components)

    def test_generic_controller_no_components(self):
        controller = GenericController(components=dict())

        self.assertDictEqual(d1=controller.components, d2=dict())
        self.assertDictEqual(d1=controller.get_fields(), d2=dict())

    def test_generic_controller_ipywidget_components(self):
        button = Button()
        check_box = Checkbox()
        components = dict(button=button, check_box=check_box)
        controller = GenericController(components=components)

        self.assertDictEqual(d1=controller.components, d2=components)
        self.assertDictEqual(d1=controller.get_fields(), d2=components)

    def test_generic_controller_ipywidget_components_make_box(self):
        button = Button()
        check_box = Checkbox()
        components = dict(button=button, check_box=check_box)
        controller = GenericController(components=components)
        box = controller.make_box(box_type=HBox)

        assert isinstance(box, HBox)
        assert box.children == (button, check_box)

    def test_generic_controller_standard_attributes(self):
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

        self.assertDictEqual(d1=controller.components, d2=components)
        self.assertDictEqual(d1=controller.get_fields(), d2=components)

    def test_generic_controller_standard_attributes_make_box(self):
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
        box = controller.make_box(box_type=HBox)

        assert isinstance(box, HBox)
        assert box.children == (button, check_box)  # Non-widget attributes were not included in the box

    def test_generic_controller_with_observers(self):
        controller = ExampleGenericControllerWithObservers()

        assert controller.check_box.value is False

        controller.button.click()

        assert controller.check_box.value is True

        controller.button.click()

        assert controller.check_box.value is False
