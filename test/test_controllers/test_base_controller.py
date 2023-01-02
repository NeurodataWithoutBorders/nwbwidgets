from hdmf.testing import TestCase

from nwbwidgets.controllers import BaseController


class ExampleSetupComponentsAbstractController(BaseController):
    def setup_components(self, components):
        pass


class ExampleSetupChildrenAbstractController(BaseController):
    def setup_children(self, box_type):
        pass


class TestBaseController(TestCase):
    def test_setup_components_abstract(self):
        with self.assertRaisesWith(
            exc_type=NotImplementedError,
            exc_msg="This Controller has not defined how to setup its components!",
        ):
            ExampleSetupChildrenAbstractController(components=list())

    def test_setup_children_abstract(self):
        with self.assertRaisesWith(
            exc_type=NotImplementedError,
            exc_msg="This Controller has not defined how to layout its children!",
        ):
            ExampleSetupComponentsAbstractController(components=list())
