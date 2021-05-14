from typing import Iterable

import numpy as np
from hdmf.common import DynamicTable
from ipywidgets import widgets, Layout, ValueWidget

from ..utils.dynamictable import group_and_sort, infer_categorical_columns
from ..utils.pynwb import robust_unique
from . import RangeController


class AbstractGroupAndSortController(widgets.VBox, ValueWidget):
    """
    Defines the abstract type for GroupAndSortController objects. These classes take in a DynamicTable objects
    and broadcast a `value` of the form
    dict(
        order=array-like(uint),
        group_inds=array-like(uint) | None,
        labels=array-like(str) | None
    )
    """

    def __init__(self, dynamic_table: DynamicTable):
        super().__init__()

        self.dynamic_table = dynamic_table
        self.nitems = len(self.dynamic_table.id)

        self.group_vals = None
        self.group_by = None
        self.group_select = None
        self.limit = None
        self.desc = False
        self.order_by = None
        self.order_vals = None
        self.window = None


class GroupAndSortController(AbstractGroupAndSortController):
    def __init__(
        self,
        dynamic_table: DynamicTable,
        group_by=None,
        window=None,
        start_discard_rows=None,
        control_order=True,
        control_limit=True,
        groups=None,
    ):
        """

        Parameters
        ----------
        dynamic_table
        group_by
        window: None or bool,
        """
        super().__init__(dynamic_table)

        self.control_order = control_order
        self.control_limit = control_limit
        start_discard_rows = [] if start_discard_rows is None else start_discard_rows
        self.keep_rows = [i for i in range(self.nitems) if i not in start_discard_rows]
        self.groups = self.get_groups() if groups is None else groups
        self.limit_bit = widgets.BoundedIntText(
            value=50, min=0, max=99999, disabled=True, layout=Layout(max_width="70px")
        )
        self.limit_bit.observe(self.limit_bit_observer)

        if control_limit:
            self.limit_cb = widgets.Checkbox(
                description="limit",
                style={"description_width": "initial"},
                disabled=True,
                indent=False,
                layout=Layout(max_width="70px"),
            )
            self.limit_cb.observe(self.limit_cb_observer)

        self.order_dd = widgets.Dropdown(
            options=[None] + list(self.groups),
            description="order by",
            layout=Layout(max_width="120px"),
            style={"description_width": "initial"},
            disabled=not len(self.groups),
        )
        self.order_dd.observe(self.order_dd_observer)

        self.ascending_dd = widgets.Dropdown(
            options=["ASC", "DESC"], disabled=True, layout=Layout(max_width="70px")
        )
        self.ascending_dd.observe(self.ascending_dd_observer)

        range_controller_max = min(30, self.nitems)
        if window is None:
            self.range_controller = RangeController(
                0,
                self.nitems,
                start_value=(0, range_controller_max),
                dtype="int",
                description="units",
                orientation="vertical",
            )
            self.range_controller.observe(self.range_controller_observer)
            self.window = self.range_controller.value
        elif window is False:
            self.window = (0, self.nitems)
            self.range_controller = widgets.HTML("")

        self.group_sm = widgets.SelectMultiple(
            layout=Layout(max_width="100px"), disabled=True, rows=1
        )
        self.group_sm.observe(self.group_sm_observer)

        if group_by is None:
            self.group_dd = widgets.Dropdown(
                options=[None] + list(self.groups),
                description="group by",
                style={"description_width": "initial"},
                layout=Layout(width="90%"),
                disabled=not len(self.groups),
            )
            self.group_dd.observe(self.group_dd_observer)
        else:
            self.group_dd = None
            self.set_group_by(group_by)

        self.children = self.get_children()
        self.layout = Layout(width="290px")
        self.update_value()

    def get_children(self):
        children = []

        if self.group_dd:
            children.append(self.group_dd)

        children.append(widgets.HBox(children=(self.group_sm, self.range_controller)))

        if self.control_limit:
            children.append(
                widgets.HBox(
                    children=(self.limit_cb, self.limit_bit),
                    layout=Layout(max_width="90%"),
                )
            )

        if self.control_order:
            children.append(
                widgets.HBox(
                    children=(self.order_dd, self.ascending_dd),
                    layout=Layout(max_width="90%"),
                )
            )

        return children

    def set_group_by(self, group_by):
        self.group_by = group_by
        self.group_vals = self.get_group_vals(by=group_by)
        group_vals = self.group_vals[self.keep_rows]
        if self.group_vals.dtype == np.float:
            group_vals = group_vals[~np.isnan(group_vals)]
        groups = np.unique(group_vals)
        self.group_sm.rows = min(len(groups), 20)
        self.group_sm.options = tuple(groups[::-1])
        self.group_sm.value = self.group_sm.options
        self.group_sm.disabled = False
        self.limit_cb.disabled = False
        self.group_and_sort()

    def group_dd_observer(self, change):
        """group dropdown observer"""
        if change["name"] == "value":
            group_by = change["new"]
            if group_by in ("None", "", None):
                self.limit_bit.disabled = True
                self.limit_cb.disabled = True
                self.group_vals = None
                self.group_by = None
                self.limit = None
                self.limit_cb.value = False
                self.group_sm.options = []
                self.group_sm.visible = False
                self.group_sm.rows = 1

                if hasattr(self.range_controller, "slider"):
                    self.range_controller.slider.max = len(self.keep_rows)
            else:
                self.set_group_by(group_by)

            self.update_value()

    def limit_bit_observer(self, change):
        """limit bounded int text observer"""
        if change["name"] == "value":
            limit = self.limit_bit.value
            self.limit = limit
            self.update_value()

    def limit_cb_observer(self, change):
        """limit checkbox observer"""
        if change["name"] == "value":
            if self.limit_cb.value and self.group_by is not None:
                self.limit_bit.disabled = False
                self.limit = self.limit_bit.value
            else:
                self.limit_bit.disabled = True
                self.limit = None
            self.update_value()

    def order_dd_observer(self, change):
        """order dropdown observer"""
        if change["name"] == "value":
            self.order_by = self.order_dd.value

            order_vals = self.get_group_vals(by=self.order_by)
            # convert to ints. This is mainly for handling strings
            _, order_vals = np.unique(order_vals, return_inverse=True)

            if self.desc:  # if descend is on, invert order.
                order_vals *= -1

            self.order_vals = order_vals

            self.ascending_dd.disabled = self.order_dd.value is None
            self.update_value()

    def ascending_dd_observer(self, change):
        """ascending dropdown observer"""
        if change["name"] == "value":
            if change["new"] == "ASC":
                self.desc = False
                self.order_vals *= -1
            else:
                self.desc = True
                self.order_vals *= -1
            self.update_value()

    def group_sm_observer(self, change):
        """group SelectMultiple observer"""
        if change["name"] == "value" and not self.group_sm.disabled:
            self.group_select = change["new"]
            value_before = self.window
            self.group_and_sort()
            if (
                hasattr(self.range_controller, "slider")
                and not self.range_controller.slider.value == value_before
            ):
                pass  # do nothing, value was updated automatically
            else:
                self.update_value()

    def range_controller_observer(self, change):
        self.window = self.range_controller.value
        self.update_value()

    def get_groups(self):
        return infer_categorical_columns(self.dynamic_table, self.keep_rows)

    def get_group_vals(self, by, units_select=None):
        """Get the values of the group_by variable

        Parameters
        ----------
        by
        units_select

        Returns
        -------

        """
        if by is None:
            return None
        elif by in self.groups:
            return self.groups[by] if units_select is None else self.groups[by][units_select]
        else:
            raise ValueError(
                "column {} not in DynamicTable {}".format(by, self.dynamic_table)
            )

    def get_orderable_cols(self):
        candidate_cols = [
            x
            for x in self.units.colnames
            if not isinstance(self.units[x][0], Iterable)
            or isinstance(self.units[x][0], str)
        ]
        return [x for x in candidate_cols if len(robust_unique(self.units[x][:])) > 1]

    def group_and_sort(self):
        if self.group_vals is None and self.order_vals is None:
            self.order_vals = np.arange(self.nitems).astype("int")

        order, group_inds, labels = group_and_sort(
            group_vals=self.group_vals,
            group_select=self.group_select,
            keep_rows=self.keep_rows,
            order_vals=self.order_vals,
            limit=self.limit,
        )

        if hasattr(self.range_controller, "slider"):
            self.range_controller.slider.max = len(order)

        # apply window
        if self.window is not None:
            order = order[self.window[0] : self.window[1]]
            if group_inds is not None:
                group_inds = group_inds[self.window[0] : self.window[1]]

        return order, group_inds, labels

    def update_value(self):

        order, group_inds, labels = self.group_and_sort()
        self.value = dict(order=order, group_inds=group_inds, labels=labels)
