from hdmf.testing import TestCase
from ipywidgets import Button, Checkbox, Dropdown, ToggleButton, HBox

from nwbwidgets.controllers import GenericController, MultiController


class ExampleGenericController1(GenericController):
    def __init__(self, components: dict):
        super().__init__(components=dict(button=Button(), toggle_button=ToggleButton()))


class ExampleGenericController2(GenericController):
    def __init__(self, components: dict):
        super().__init__(components=dict(dropdown=Dropdown(), check_box=Checkbox()))


class ExampleMultiController1(MultiController):
    def __init__(self, components: dict):
        super().__init__(
            components=dict(
                example_controller_1=ExampleGenericController1(), example_controller_2=ExampleGenericController2()
            )
        )


class ExampleMultiController2(MultiController):
    def __init__(self, components: dict):
        super().__init__(
            components=dict(
                example_controller_1=ExampleGenericController1(), example_controller_2=ExampleGenericController2()
            )
        )


class TestMultiController(TestCase):
    def test_name_collision(self):
        pass  # TODO, simlar to generic tests but for nested controllers as well

    def test_multi_controller_no_components(self):
        controller = GenericController(components=dict(button=Button(), check_box=Checkbox()))

        self.assertDictEqual(d1=controller.components, d2=dict())
        self.assertDictEqual(d1=controller.get_fields(), d2=dict())

    def test_multi_controller_ipywidget_components(self):
        button = Button()
        check_box = Checkbox()
        components = dict(button=button, check_box=check_box)
        controller = GenericController(components=components)

        self.assertDictEqual(d1=controller.components, d2=components)
        self.assertDictEqual(d1=controller.get_fields(), d2=components)

    def test_multi_controller_ipywidget_components_make_box(self):
        button = Button()
        check_box = Checkbox()
        components = dict(button=button, check_box=check_box)
        controller = GenericController(components=components)

        box = controller.make_box(box_type=HBox)
        assert box == HBox(children=(button, check_box))

    def test_multi_controller_standard_attributes(self):
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

    def test_multi_controller_standard_attributes_make_box(self):
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
        assert box == HBox(children=(button, check_box))  # Non-widget attributes were not included in the box

    # TODO: add a test with observers

    # def test_base_controller_other_controller_component(self):
    #     button = Button()
    #     secondary_controller = BaseController(components=dict(test_component=button))

    #     main_controller = BaseController(components=dict(other_controller=secondary_controller))

    #     self.assertDictEqual(d1=main_controller.components, d2=dict(other_controller=secondary_controller))
    #     self.assertDictEqual(d1=main_controller.get_fields(), d2=dict(other_controller=secondary_controller))

    # def test_base_controller_other_controller_component_with_attribute(self):
    #     def SecondaryController(BaseController):
    #         def __init__(self):
    #             button = Button()

    #             self.some_attribute = 5  # This attribute is then mutable by observers defined in this controller...

    #             super().__init__(components=dict(some_button=button))  # But is not technically a full 'component'...

    #     secondary_controller = SecondaryController()

    #     main_controller = BaseController(components=dict(test_component=secondary_controller))

    #     self.assertDictEqual(d1=main_controller.components, d2=dict(test_component=secondary_controller))
    #     self.assertDictEqual(
    #         d1=main_controller.get_fields(), d2=dict(test_component=secondary_controller, some_attribute=5)
    #     )

    # def test_base_controller_nested_controller_component(self):
    #     def SecondaryController(BaseController):
    #         def __init__(self):
    #             button = Button()

    #             self.some_attribute = 5  # This attribute is then mutable by observers defined in this controller...

    #             super().__init__(components=dict(some_button=button))  # But is not technically a full 'component'...

    #     def TertiaryController(BaseController):
    #         def __init__(self):
    #             check_box = Checkbox()

    #             self.other_attribute = "test"

    #             super().__init__(components=dict(check_box=check_box))

    #     tertiary_controller = TertiaryController()
    #     secondary_controller = SecondaryController(components=dict(tertiary_controller=tertiary_controller))

    #     main_controller = BaseController(components=dict(test_component=secondary_controller))

    #     self.assertDictEqual(
    #         d1=main_controller.components, d2=dict(test_component=dict(tertiary_controller=tertiary_controller))
    #     )
    #     self.assertDictEqual(
    #         d1=main_controller.get_fields(),
    #         d2=dict(
    #             test_component=secondary_controller,
    #             some_attribute=5,
    #             tertiary_controller=tertiary_controller,
    #             other_attribute="test",
    #         ),
    #     )
