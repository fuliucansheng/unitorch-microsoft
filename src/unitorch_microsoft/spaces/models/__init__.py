# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch_microsoft import cached_path
from unitorch_microsoft.spaces import (
    create_element,
    create_row,
    create_column,
    create_flex_layout,
    create_blocks,
    create_toper_menus,
    create_footer,
    create_dashboard_card,
    create_card,
    create_dashboard_cards_group,
    create_cards_group,
)

config_path = cached_path("spaces/config.ini")
config = CoreConfigureParser(config_path)

generation = [
    GenericOutputs(
        title="Query | Keyword",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/qk",
    ),
    GenericOutputs(
        title="Text | Image",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/ti",
    ),
]

selection = [
    GenericOutputs(
        title="Query | Keyword",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/qk",
    ),
    GenericOutputs(
        title="Text | Image",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/ti",
    ),
    GenericOutputs(
        title="Query | Keyword",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/qk",
    ),
    GenericOutputs(
        title="Text | Image",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/ti",
    ),
    GenericOutputs(
        title="Query | Keyword",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/qk",
    ),
    GenericOutputs(
        title="Text | Image",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/ti",
    ),
]

relevance = [
    GenericOutputs(
        title="Query | Keyword",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/qk",
    ),
    GenericOutputs(
        title="Text | Image",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/ti",
    ),
    GenericOutputs(
        title="Query | Keyword",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/qk",
    ),
    GenericOutputs(
        title="Text | Image",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/ti",
    ),
    GenericOutputs(
        title="Query | Keyword",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/qk",
    ),
    GenericOutputs(
        title="Text | Image",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/ti",
    ),
]

ranking = [
    GenericOutputs(
        title="Query | Keyword",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/qk",
    ),
    GenericOutputs(
        title="Text | Image",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/ti",
    ),
    GenericOutputs(
        title="Query | Keyword",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/qk",
    ),
    GenericOutputs(
        title="Text | Image",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/ti",
    ),
    GenericOutputs(
        title="Query | Keyword",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/qk",
    ),
    GenericOutputs(
        title="Text | Image",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/models/ti",
    ),
]


def create_models_page():
    toper_menus = create_toper_menus()
    dashboard_group = create_dashboard_cards_group(
        "🌐 Overview",
        generation,
    )
    generation_group = create_cards_group(
        "Generation",
        generation,
    )
    selection_group = create_cards_group(
        "Selection",
        selection,
    )
    relevance_group = create_cards_group(
        "Relevance",
        relevance,
    )
    ranking_group = create_cards_group(
        "Ranking",
        ranking,
    )
    footer = create_footer()
    return create_blocks(
        toper_menus,
        dashboard_group,
        generation_group,
        selection_group,
        relevance_group,
        ranking_group,
        footer,
    )


models_page = create_models_page()
models_page.title = "Ads Spaces | Models Home"

# pages = {}
# for page in generation + selection + relevance + ranking:
#     page = ExampleWebUI(config, page.title, page.desc, page.link).iface
#     page.title = "Ads Spaces | Models - " + page.title
#     pages[page.link] = page


models_routers = {
    "/models": models_page,
}
