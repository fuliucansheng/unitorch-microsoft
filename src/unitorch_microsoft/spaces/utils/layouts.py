# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import re
import socket
import requests
import tempfile
import hashlib
import subprocess
import pandas as pd
import gradio as gr
from PIL import Image
from collections import Counter, defaultdict
from torch.hub import download_url_to_file
from unitorch import get_temp_home
from unitorch.cli import CoreConfigureParser
from unitorch.cli.webuis import (
    create_element,
    create_accordion,
    create_row,
    create_column,
    create_tab,
    create_tabs,
    create_flex_layout,
    create_group,
    create_tab,
    create_tabs,
    create_blocks,
)
from unitorch_microsoft.spaces.utils import random_bg_color


def create_toper_menus():
    left = create_element(
        "markdown",
        "<a class='ut-ms-a-none' href='/'><img src='/_publics/logo.svg' /><span style='font-size: 2em; font-weight: 700'>Ads Spaces</span></a>",
    )
    middle = create_element(
        "markdown", "<div></div>", elem_classes="ut-ms-place-holder-block"
    )
    menus = [
        create_element(
            "markdown",
            "<a class='ut-ms-a-none' style='font-size:1.2em; color:black;' href='/picasso'><span>💡 Picasso</span></a>",
        ),
        create_element(
            "markdown",
            "<a class='ut-ms-a-none' style='font-size:1.2em; color:black;' href='/models'><span>🔥 Models</span></a>",
        ),
        # create_element(
        #     "markdown",
        #     "<a class='ut-ms-a-none' style='font-size:1.2em; color:black;' href='/prompts'><span>🎉 Prompts</span></a>",
        # ),
        create_element(
            "markdown",
            "<a class='ut-ms-a-none' style='font-size:1.2em; color:black;' href='/demos'><span>🚀 Demos</span></a>",
        ),
        create_element(
            "markdown",
            "<a class='ut-ms-a-none' style='font-size:1.2em; color:black;' href='/docs'><span>📚 Docs</span></a>",
        ),
    ]
    right = create_row(*menus, elem_classes="ut-ms-vh-center-bottom ut-ms-toper-menus")
    return create_row(
        create_column(left, scale=1),
        create_column(
            middle, scale=1, elem_classes="ut-bg-transparent ut-ms-place-holder-block"
        ),
        create_column(right, scale=1),
        elem_classes="ut-bg-transparent ut-ms-toper ut-ms-vh-center-bottom",
    )


def create_footer():
    left = create_element(
        "markdown",
        "<a class='ut-ms-a-none' href='/'><img src='/_publics/logo.svg' /><span style='font-size: 1em;'>© AdsPlus Team<span></a>",
    )
    middle = create_element(
        "markdown",
        "<div style='width: 100%; display: flex; justify-content: center'>🌴 Powered By &nbsp;<a class='ut-ms-a-none' style='color: inherit' href='https://github.com/fuliucansheng/unitorch'>unitorch</a> &nbsp;</div>",
        elem_classes="ut-bg-transparent",
    )
    right = create_element(
        "markdown",
        # """
        # <div style='display:flex; justify-content: flex-end; padding-right: 20px;'>
        #   <a class='ut-ms-a-none' style='color:black' href='#'>Github</a>
        #   &nbsp;·&nbsp;
        #   <a class='ut-ms-a-none' style='color:black' href='#'>Github</a>
        # </div>
        # """,
        """
        <div style='display:flex; justify-content: flex-end; padding-right: 20px;'>
          <a class='ut-ms-a-none' style='color:inherit' href='#'>🍀&nbsp;Github</a>
        </div>
        """,
    )
    return create_row(
        create_column(left, scale=1),
        create_column(
            middle, scale=2, elem_classes="ut-bg-transparent ut-ms-place-holder-block"
        ),
        create_column(right, scale=1),
        elem_classes="ut-bg-transparent ut-ms-footer ut-ms-vh-center-bottom",
    )


def create_dashboard_card(title, desc, link, elem_classes="ut-ms-dashboard-card-item"):
    bg_color = random_bg_color(title + desc)
    html = f"""
      <a  class='ut-ms-a-none' style='display:block' href='{link}'>
        <div class='{elem_classes} {bg_color}'>
          <p style='font-weight: 600; font-size: 1.5em;'>{title}</p>
          <p style='font-size: 1.2em;'>{desc}</p>
        </div>
      </a>
    """
    return create_column(
        create_element("markdown", html),
        elem_classes="ut-ms-dashboard-card-item ut-bg-transparent",
    )


def create_card(title, desc, link, elem_classes="ut-ms-card-item"):
    bg_color = random_bg_color(title + desc)
    html = f"""
      <a  class='ut-ms-a-none' style='display:block' href='{link}'>
        <div class='{elem_classes} {bg_color}'>
          <p style='font-weight: 600; font-size: 1em;'>{title}</p>
          <p style='font-size: 0.85em;'>{desc}</p>
        </div>
      </a>
    """
    return create_column(
        create_element("markdown", html),
        elem_classes="ut-ms-card-item ut-bg-transparent",
    )


def create_dashboard_cards_group(header, cards, elem_classes=None):
    header = create_element(
        "markdown",
        f"# <div style='margin-top:10px'>{header}</div>",
    )
    cards = [
        create_dashboard_card(
            card.title, card.desc, card.link, elem_classes="ut-ms-dashboard-card-item"
        )
        for card in cards
    ]
    layout = create_flex_layout(*cards, num_per_row=2, do_padding=True)
    return create_column(
        header,
        layout,
        elem_classes=f"ut-ms-dashboard-card-group ut-bg-transparent {elem_classes}",
    )


def create_cards_group(header, cards, elem_classes=None):
    header = create_element(
        "markdown",
        f"# <div style='margin-top:10px'>{header}</div>",
    )
    cards = [
        create_card(card.title, card.desc, card.link, elem_classes="ut-ms-card-item")
        for card in cards
    ]
    layout = create_row(*cards, elem_classes="ut-ms-card-item ut-bg-transparent")
    return create_column(
        header,
        layout,
        elem_classes=f"ut-ms-card-group ut-bg-transparent {elem_classes}",
    )
