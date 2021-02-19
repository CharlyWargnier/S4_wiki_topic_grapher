import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

import networkx as nx
from networkx.readwrite import json_graph
from pyvis import network as net

import matplotlib.pyplot as plt
import seaborn as sns

from bs4 import BeautifulSoup
import pywikibot
import math
import os
import re
import requests
import tempfile
import validators

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from google.cloud import language_v1
from google.cloud.language_v1 import enums


st.set_page_config(
    page_title="Wiki Topic Grapher",
    page_icon="favicon.ico",
)


def _max_width_():
    max_width_str = f"max-width: 1500px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()


c30, c31, c32 = st.beta_columns([1, 3.3, 3])


with c30:
    st.markdown("###")
    st.image("wikilogo.png", width=520)
    st.header("")

with c32:
    st.markdown("#")
    st.text("")
    st.text("")
    st.markdown(
        "###### Original script by [JR Oakes](https://twitter.com/jroakes) - Ported to [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/)&nbsp, with :heart: by [DataChaz](https://twitter.com/DataChaz) &nbsp [![this is an image link](https://i.imgur.com/thJhzOO.png)](https://www.buymeacoffee.com/cwar05)"
    )


with st.beta_expander("ℹ️  - About this app ", expanded=True):
    st.write(
        """  

-   Wiki Topic Grapher leverages the power of [Google Natural Language API] (https://cloud.google.com/natural-language) to recursively retrieve entity relationships from any Wikipedia seed topic! 🔥
-   Get a network graph of these connected entities, save the graph as jpg or export the results ordered by salience to CSV!
-   The tool is still in Beta, with possible rough edges! [![Gitter](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/DataChaz/WikiTopic) for bug report, questions, or suggestions.
-   Kudos to JR Oakes for the original script - [buy the man a 🍺 here!](https://www.paypal.com/paypalme/codeseo)
-   This app is free. If it's useful to you, you can [buy me a ☕](https://www.buymeacoffee.com/cwar05) to support my work! 🙏


	    """
    )

    st.markdown("---")


with st.beta_expander("🛠️ - How to use it ", expanded=False):

    st.markdown(
        """  
- Wiki Topic Grapher takes the top entities for each Wikipedia URL and follows those entities according to the specified limit and depth parameters
- Here's a [neat chart](https://i.imgur.com/wZOU1wh.png) explaining how it all works"""
    )

    st.markdown("---")

    st.markdown(
        """  

**URL:**

- Paste a Wikipedia URL
- Make sure the URL belongs to https://en.wikipedia.org/
- Only English is currently supported. More languages to come! :)

_

**Topic:**

- Select "Topic" via the left-hand toggle and type your keyword
- It will return the closest matching Wikipedia page for that given string
- Use that method with caution as currently there's no way to get the related page before calling the API.
- So this can be costly if the selected page has tons of content!

_

**Depth**:
- The maximum number of entities to pull for each Wikipedia page
- Depth 1 or 2 are the recommended settings
- Depth 3 and above work yet it may not be usable nor legible!

_

**Limit**:
- The max number of entities to pull for each page

	    """
    )

    st.markdown("---")

with st.beta_expander("🔎- SEO use cases ", expanded=False):
    st.write(
        """  

-   Research any topic then get entity associations that exist from that seed topic
-   Map out these related entities & alternative lexical fields with your product, service or brand
-   Find how well you've covered a specific topic on your website
-   Differentiate pages on your website!

	    """
    )

    st.markdown("---")


with st.beta_expander("🧰 - Stack + To-Do's", expanded=False):

    st.markdown("")

    st.write(
        """  
** Stack **

-   100% Python! 🐍🔥
-   [Google Natural Language API](https://cloud.google.com/natural-language)
-   [PyWikibot](https://www.mediawiki.org/wiki/Manual:Pywikibot)
-   [Networkx](https://networkx.org/)
-   [Streamlit](https://www.streamlit.io/)
-   [Streamlit Components](https://www.streamlit.io/components)"""
    )

    st.markdown("")

    st.write(
        """  

** To-Do's **

-   Add a budget estimator to estimate Google Cloud Language API costs
-   Add a multilingual option (currently English only)  
-   Add on-the-fly physics controls to the network graph 
-   Exception handling is still pretty broad at the moment and could be improved

	    """
    )

    st.markdown("---")

st.markdown("## **① Upload your Google NLP key **")
with st.beta_expander("ℹ️ - How to create your credentials?", expanded=False):

    st.write(
        """
	          
      - In the [Cloud Console](https://console.cloud.google.com/), go to the _'Create Service Account Key'_  page
      - From the *Service account list*, select  _'New service account'_
      - In the *Service account name* field, enter a name
      - From the *Role list*, select  _'Project > Owner'_
      - Click create, then download your JSON key
      - Upload it (or drag and drop it) in the grey box below 👇

	    """
    )
    st.markdown("---")


# Pywikibot needs a config file
pywikibot_config = r"""# -*- coding: utf-8  -*-
mylang = 'en'
family = 'wikipedia'
usernames['wikipedia']['en'] = 'test'"""

with open("user-config.py", "w", encoding="utf-8") as f:
    f.write(pywikibot_config)

c3, c4 = st.beta_columns(2)

with c3:
    try:
        uploaded_file = st.file_uploader("", type="json")
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            fp.write(uploaded_file.getvalue())
        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = fp.name
            with open(fp.name, "rb") as a:
                client = language.LanguageServiceClient.from_service_account_json(
                    fp.name
                )

        finally:
            if os.path.isfile(fp.name):
                os.unlink(fp.name)

    except AttributeError:

        print("wait")

with c4:
    st.markdown("###")
    c = st.beta_container()
    if uploaded_file:
        st.success("✅ Nice! Your credentials are uploaded!")


def google_nlp_entities(
    input,
    input_type="html",
    result_type="all",
    limit=10,
    invalid_types=["OTHER", "NUMBER", "DATE"],
):

    """
    Loads HTML or text from a URL and passes to the Google NLP API
    Parameters:
        * input: HTML or Plain Text to send to the Google Language API
        * input_type: Either `html` or `text` (string)
        * result_type: Either `all`(pull all entities) or `wikipedia` (only pull entities with Wikipedia pages)
        * limit: Limits the number of results to this number sorted, decending, by salience.
        * invalid_types: A list of entity types to exclude.
    Returns:
        List of entities in format [{'name':<name>,'type':<type>,'salience':<salience>, 'wikipedia': <wikipedia url - optional>}]
    """

    def get_type(type):
        return client.enums.Entity.Type(d.type).name

    if not input:
        print("No input content found.")
        return None

    if input_type == "html":
        doc_type = language.enums.Document.Type.HTML
    else:
        doc_type = language.enums.Document.Type.PLAIN_TEXT

    document = types.Document(content=input, type=doc_type)

    features = {"extract_entities": True}

    try:
        response = client.annotate_text(
            document=document, features=features, timeout=20
        )
    except Exception as e:
        print("Error with language API: ", re.sub(r"\(.*$", "", str(e)))
        return []

    used = []
    results = []
    for d in response.entities:

        if limit and len(results) >= limit:
            break

        if get_type(d.type) not in invalid_types and d.name not in used:

            data = {
                "name": d.name,
                "type": client.enums.Entity.Type(d.type).name,
                "salience": d.salience,
            }
            if result_type is "wikipedia":
                if "wikipedia_url" in d.metadata:
                    data["wikipedia"] = d.metadata["wikipedia_url"]
                    results.append(data)
            else:
                results.append(data)

            used.append(d.name)

    return results


def load_page_title(url):
    """
    Returns the <title> given a URL.
    Parameters:
        * url: URL (string)
    Returns:
       Inner text of <title> (string)
    """
    soup = BeautifulSoup(requests.get(url).text)
    return soup.title.text


@st.cache(allow_output_mutation=True, show_spinner=False)
def html_to_text(html, target_elements=None):
    """
    Transforms HTML to clean text
    Parameters:
        * html: HTML from a web page (str)
        * target_elements: Elements like `div` or `p` to target pulling text from. (optional) (string)
    Returns:
        Text (string)
    """
    soup = BeautifulSoup(html)

    for script in soup(
        ["script", "style"]
    ):  # remove all javascript and stylesheet code
        script.extract()

    targets = []

    if target_elements:
        targets = soup.find_all(target_elements)

    if target_elements and len(targets) > 3:
        text = " ".join([t.text for t in targets])
    else:
        text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = "\n".join(chunk for chunk in chunks if chunk)
    return text


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_text_from_url(url, **data):

    """
    Loads html from a URL
    Parameters:
        * url: url of page to load (str)
        * timeout: request timeout in seconds (int) default: 20
    Returns:
        HTML (str)
    """

    timeout = data.get("timeout", 20)

    results = []

    try:
        # print("Extracting HTML from: {}".format(url))
        response = requests.get(
            url,
            headers={
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:68.0) Gecko/20100101 Firefox/68.0"
            },
            timeout=timeout,
        )

        text = response.text
        status = response.status_code

        if status == 200 and len(text) > 0:
            return text
        else:
            print("Incorrect status returned: ", status)

        return None

    except Exception as e:
        print("Problem with url: {0}.".format(url))
        return None


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_wikipedia_url(query):
    """
    Finds the closest matching Wikipedia page for a given string.
    Parameters:
        * query: Query to search Wikipedia with. (string)
    Returns:
       The top matching URL for the query.  Follows redirects (string)
    """
    sitew = pywikibot.Site("en", "wikipedia")
    result = None
    print("looking up:", query)
    search = sitew.search(
        query, where="title", get_redirects=True, total=1, content=False, namespaces="0"
    )
    for page in search:
        if page.isRedirectPage():
            page = page.getRedirectTarget()
        result = page.full_url()
        break

    return result


@st.cache(allow_output_mutation=True, show_spinner=False)
def recurse_entities(
    input_data, entity_results=[], G=nx.Graph(), current_depth=0, depth=2, limit=3
):
    """
    Recursively finds entities of connected Wikipedia topics by taking the top entities
    for each page and following those entities up to the specified depth
    Parameters:
        * input_data: A topic or URL.  If topic, finds the closes matching Wikipedia start page.
                      If URL, starts with the top enetities of that page. (string)
        * depth: Max recursion depth (integer)
        * limit: The max number of entities to pull for each page. (integer)
    Returns:
       A tuple of:
        * entity_results: List of dictionaries of found entities.
        * G: Networkx graph of entities.
    """
    if isinstance(input_data, str):
        # Starting fresh.  Make sure variables are fresh.
        entity_results = []
        G = nx.Graph()
        current_depth = 0
        if not validators.url(input_data):
            input_data = get_wikipedia_url(input_data)
            if not input_data:
                print("No Wikipedia URL Found.")
                return None, None
            else:
                print("Wikipedia URL: ", input_data)
            name = load_page_title(input_data).split("-")[0].strip()
        else:
            name = load_page_title(input_data)
        input_data = (
            [
                {
                    "name": name.title(),
                    "type": "START",
                    "salience": 0.0,
                    "wikipedia": input_data,
                }
            ]
            if input_data
            else []
        )

    # Regex for wikipedia terms to not bias entities returned
    subs = r"(wikipedia|wikimedia|wikitext|mediawiki|wikibase)"

    for d in input_data:
        url = d["wikipedia"]
        name = d["name"]

        print(
            "   " * current_depth + "Level: {0} Name: {1}".format(current_depth, name)
        )

        html = load_text_from_url(url)

        # html_to_text will default to all text if < 4 `p` elements found.
        if "wikipedia.org" in url:
            html = html_to_text(html, target_elements="p")
        else:
            html = html_to_text(html)

        # Kill brutally wikipedia terms.
        html = re.sub(subs, "", html, flags=re.IGNORECASE)

        results = [
            r
            for r in google_nlp_entities(
                html, input_type="text", limit=None, result_type="wikipedia"
            )
            if "wiki" not in r["name"].lower() and not G.has_node(r["name"])
        ][:limit]
        _ = [G.add_edge(name, r["name"]) for r in results]
        entity_results.extend(results)

        new_depth = int(current_depth + 1)
        if results and new_depth <= depth:
            recurse_entities(results, entity_results, G, new_depth, depth, limit)

    if current_depth == 0:
        return entity_results, G


@st.cache(allow_output_mutation=True, show_spinner=False)
def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):

    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(G))
            )  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(
        G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None
    ):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def plot_entity_branches(G, w=10, h=10, c=1, font_size=14, filename=None):
    """
    Given a networkx graph, builds a recursive tree graph

    Parameters:
        * G: Networkx graph of entities.
        * w: Width of output plot
        * h: height of output plot
        * c: Circle percentage (float) 0.5 is a semi-circle. Range: 0.1-1.0
        * font_size: Font Size of labels (integer)
        * filename: Filename for the saved plot.  Optional (string)
    Returns:
       Nothing. Plots a graph

    """
    start = list(G.nodes)[0]
    G = nx.bfs_tree(G, start)
    plt.figure(figsize=(w, h))
    pos = hierarchy_pos(G, start, width=float(2 * c) * math.pi, xcenter=0)
    new_pos = {
        u: (r * math.sin(theta), r * math.cos(theta)) for u, (theta, r) in pos.items()
    }
    nx.draw(
        G,
        pos=new_pos,
        alpha=0.8,
        node_size=25,
        with_labels=True,
        font_size=font_size,
        edge_color="gray",
    )
    nx.draw_networkx_nodes(
        G, pos=new_pos, nodelist=[start], node_color="blue", node_size=500
    )

    if filename:
        plt.savefig("{0}/{1}".format("images", filename))


st.set_option("deprecation.showPyplotGlobalUse", False)

st.markdown("## **② Choose a URL or a topic **")

with st.beta_expander("ℹ️ - How Google Cloud pricing works ", expanded=False):

    st.write(
        """
        - Your usage of the Google Natural Language API is calculated in terms
       of "units"
       - Each document sent to the API for analysis is at least one unit
       - Documents that have more than 1,000 Unicode characters are considered as multiple units (1 unit per 1,000 characters)
       -   More info about pricing on [Google's website](https://cloud.google.com/natural-language/pricing)

	    """
    )

    st.markdown("---")

st.text("")

try:

    c10, c0, c8, c1, c2, c3, c4, c5, c6 = st.beta_columns(
        [0.10, 0.50, 0.10, 8, 0.10, 1.5, 0.10, 1.5, 0.10]
    )

    with c0:
        st.text("")
        toggle = st.select_slider("", options=("URL", "Tpc"))

    with c1:

        from re import search

        substring = "http://|https://"

        if toggle == "Tpc":
            keyword = st.text_input(
                "Enter a topic. (Returns the closest matching Wikipedia page for a given string)",
                key=1,
            )
            if keyword:
                if search(substring, keyword):
                    st.warning(
                        "⚠️ Seems like you&#39re trying to paste a URL. Switch to &#39URL&#39 mode?"
                    )
                    st.stop()
                else:
                    st.markdown('Keyword is "' + str(keyword) + '"')

        elif toggle == "URL":

            keyword = st.text_input(
                "Enter a Wikipedia URL",
                key=2,
            )

            if keyword:
                if search(substring, keyword):
                    st.markdown('URL is "' + str(keyword) + '"')
                else:
                    st.warning(
                        "⚠️ Please check the URL format as it's invalid. It needs to start with http:// or https://. If you wanted to paste a keyword, switch to 'Topic' mode."
                    )
                    st.stop()

    with c3:
        depth = st.number_input(
            "Depth", step=1, value=1, min_value=1, max_value=3, key=1
        )

    with c5:
        limit = st.number_input(
            "Limit", step=1, value=1, min_value=1, max_value=3, key=2
        )

    c3, c4 = st.beta_columns(2)

    with c3:
        st.text("")
        st.text("")
        cButton = st.beta_container()

    with c4:
        st.text("")
        c30 = st.beta_container()

    button1 = cButton.button("✨ Happy with costs, get me the data!")

    if not button1 and not uploaded_file:
        st.stop()
    elif not button1 and uploaded_file:
        st.stop()
    elif button1 and not uploaded_file:
        c.warning("◀️ Add credentials 1st")
        st.stop()
    else:
        pass

    if button1:

        import time

        latest_iteration = st.empty()
        bar = st.progress(0)

        for i in range(100):
            latest_iteration.markdown(f"Sending your request ({i+1} % Completed)")
            bar.progress(i + 1)
            time.sleep(0.05)

    data, G = recurse_entities(keyword, depth=depth, limit=limit)

    st.markdown("## **③ Check results! ✨**")

    st.text("")

    g4 = net.Network(
        directed=True,
        heading="",
        height="800px",
        width="800px",
        notebook=True,
    )

    c1, c2, c3 = st.beta_columns([1, 3, 2])

    with c2:
        g4.from_nx(G)
        g4.show("wikiOutput.html")
        HtmlFile = open("wikiOutput.html", "r")
        source_code = HtmlFile.read()
        components.html(source_code, height=1000, width=1000)

    c30, c31, c32 = st.beta_columns(3)

    with c30:
        c1 = st.beta_container()
    with c31:
        c2 = st.beta_container()

    cm = sns.light_palette("green", as_cmap=True)
    df = pd.DataFrame(data).sort_values(by="salience", ascending=False)
    df = df.reset_index()
    df.index += 1
    df = df.drop(["index"], axis=1)
    format_dictionary = {
        "salience": "{:.1%}",
    }
    dfStyled = df.style.background_gradient(cmap=cm)
    dfStyled2 = dfStyled.format(format_dictionary)
    st.table(dfStyled2)

    try:
        import base64

        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="listViewExport.csv">** - Download data to CSV 🎁 **</a>'
        c1.markdown(href, unsafe_allow_html=True)
    except NameError:
        print("wait")

except Exception as e:

    st.warning(
        f"""
            🤔 ** Snap! **
            have you checked that:
             -  The credentials JSON file you have added is valid?
             -  Google Cloud's billing is enabled?
             -  The URL you typed is a valid Wikipedia URL (that is, if you selected the "URL" option)?            

            If this keeps happening -> [![Gitter](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/DataChaz/WikiTopic)
            
            """
    )
