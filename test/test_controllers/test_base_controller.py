from hdmf.testing import TestCase
from ipywidgets import Button, Checkbox, HBox

from nwbwidgets.controllers import BaseController


# def TestController(BaseController):
#     def __init__(self, components: dict):
#         super().__init__(components=components)

#     def make_box(self, box_type=HBox) -> HBox:
#         return box_type(children=tuple(self.components.values()))


class TestBaseController(TestCase):
    @classmethod
    def setUpClass(cls):
        class TestController(BaseController):
            def __init__(self, components: dict):
                super().__init__(components=components)

            def make_box(self, box_type=HBox) -> HBox:
                return box_type(children=tuple(self.components.values()))

        cls.TestController = TestController

    def test_name_collision(self):
        pass

    def test_to_box_not_implemented(self):
        pass

    def test_base_controller_no_components(self):
        controller = self.TestController(components=dict())

        self.assertDictEqual(d1=controller.components, d2=dict())
        self.assertDictEqual(d1=controller.get_fields(), d2=dict())

    def test_base_controller_ipywidget_component(self):
        button = Button()

        controller = self.TestController(components=dict(test_component=button))

        self.assertDictEqual(d1=controller.components, d2=dict(test_component=button))
        self.assertDictEqual(d1=controller.get_fields(), d2=dict(test_component=button))

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
