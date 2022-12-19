from hdmf.testing import TestCase

from nwbwidgets.controllers import BaseController


class ExampleSetupComponentsAbstractController(BaseController):
    def setup_components(self, components):
        pass


class ExampleMakeBoxAbstractController(BaseController):
    def make_box(self, box_type):
        pass


class TestBaseController(TestCase):
    def test_setup_components_abstract(self):
        with self.assertRaisesWith(
            exc_type=TypeError,
            exc_msg=(
                "Can't instantiate abstract class BaseController with abstract methods make_box, setup_components"
            ),
        ):
            BaseController(components=list())

    def test_setup_controller_abstract(self):
        with self.assertRaisesWith(
            exc_type=TypeError,
            exc_msg=(
                "Can't instantiate abstract class ExampleMakeBoxAbstractController "
                "with abstract method setup_components"
            ),
        ):
            ExampleMakeBoxAbstractController(components=list())

    def test_make_box_abstract(self):
        with self.assertRaisesWith(
            exc_type=TypeError,
            exc_msg=(
                "Can't instantiate abstract class ExampleSetupComponentsAbstractController "
                "with abstract method make_box"
            ),
        ):
            ExampleSetupComponentsAbstractController(components=list())
