from hdmf.testing import TestCase

from ipywidgets import Button, Checkbox

from nwbwidgets.controllers import BaseController


class TestBaseController(TestCase):
    def test_setup_attributes_abstract(self):
        class ExampleSetupChildrenAbstractController(BaseController):
            def setup_children(self):
                pass

        with self.assertRaisesWith(
            exc_type=NotImplementedError,
            exc_msg="This Controller has not defined how to setup its attributes!",
        ):
            ExampleSetupChildrenAbstractController()

    def test_setup_children_abstract(self):
        class ExampleSetupComponentsAbstractController(BaseController):
            def setup_attributes(self):
                pass

        with self.assertRaisesWith(
            exc_type=NotImplementedError,
            exc_msg="This Controller has not defined how to layout its children!",
        ):
            ExampleSetupComponentsAbstractController()

    def test_generic_controller_no_components(self):
        class EmptyController:
            def setup_attribute(self):
                pass

            def setup_children(self):
                self.children = ()

        controller = EmptyController()

        self.assertDictEqual(d1=controller.get_fields(), d2=dict())
        self.assertTupleEqual(tuple1=controller.children, tuple2=())

    def test_generic_controller_ipywidget_components(self):
        class BasicController:
            def setup_attribute(self):
                self.button = Button()
                self.check_box = Checkbox()

            def setup_children(self):
                self.children = (self.button, self.check_box)

        controller = BasicController()

        assert isinstance(controller.button, Button)
        assert isinstance(controller.check_box, Checkbox)

        assert len(self.children) == 2
        assert isinstance(controller.children[0], Button)
        assert isinstance(controller.children[1], Checkbox)

        fields = controller.get_fields()
        assert len(fields) == 2
        assert "button" in fields
        assert "check_box" in fields


# TODO: update tests below here
# def test_generic_controller_standard_attributes(self):
#     """Non-widget attributes were not included in the children of the box."""
#     button = Button()
#     check_box = Checkbox()
#     components = dict(
#         button=button,
#         check_box=check_box,
#         some_integer=1,
#         some_string="test",
#         some_float=1.23,
#         some_list=[1, 2, 3],
#         some_dict=dict(a=5, b=6, c=7),
#     )
#     controller = GenericController(components=components)

#     expected_components = dict(button=button, check_box=check_box)
#     expected_fields = dict(
#         button=button,
#         check_box=check_box,
#         some_integer=1,
#         some_string="test",
#         some_float=1.23,
#         some_list=[1, 2, 3],
#         some_dict=dict(a=5, b=6, c=7),
#     )
#     expected_children = (button, check_box)
#     self.assertDictEqual(d1=controller.components, d2=expected_components)
#     self.assertDictEqual(d1=controller.get_fields(), d2=expected_fields)
#     assert controller.children == expected_children
#     self.assertTupleEqual(tuple1=controller.children, tuple2=expected_children)

# def test_generic_controller_with_observers(self):
#     class ExampleControllerWithObservers(BaseController):
#         def __init__(self):
#             super().__init__(components=dict(button=Button(), check_box=Checkbox()))

#         def check_box_on_button_click(self, change):
#             if self.check_box.value is True:
#                 self.check_box.value = False
#             elif self.check_box.value is False:
#                 self.check_box.value = True

#         def setup_observers(self):
#             self.button.on_click(self.check_box_on_button_click)

#     controller = ExampleControllerWithObservers()

#     assert controller.check_box.value is False

#     controller.button.click()

#     assert controller.check_box.value is True

#     controller.button.click()

#     assert controller.check_box.value is False
