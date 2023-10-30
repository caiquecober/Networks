import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
from IPython.core.display import display, HTML
from plotly.offline import init_notebook_mode


def assign_colour(correlation):
    if correlation <= 0:
        return "#ffa09b"  # red
    else:
        return "#9eccb7"  # green


def assign_thickness(correlation, benchmark_thickness=2, scaling_factor=3):
    return benchmark_thickness * abs(correlation) ** scaling_factor


def assign_node_size(degree, scaling_factor=50):
    return degree * scaling_factor

def convert_rankings_to_string(ranking):
    """
    Concatenate list of nodes and correlations into a single html
    string (required format for the plotly tooltip)

    Inserts html "<br>" inbetween each item in order to add a new
    line in the tooltip
    """
    s = ""
    for r in ranking:
        s += r + "<br>"
    return s


def calculate_stats(returns):
    """calculate annualised returns and volatility for all ETFs

    Returns:
        tuple: Outputs the annualised volatility and returns as a list of
            floats (for use in assigning node colours and sizes) and also
            as a lists of formatted strings to be used in the tool tips.
    """

    # log returns are additive, 252 trading days
    annualized_returns = list(np.mean(returns) * 252 * 100)

    annualized_volatility = [
        np.std(returns[col] * 100) * (252 ** 0.5) for col in list(returns.columns)
    ]

    # create string for tooltip
    annualized_volatility_2dp = [
        "Annualized Volatility: " "%.1f" % r + "%" for r in annualized_volatility
    ]
    annualized_returns_2dp = [
        "Annualized Returns: " "%.1f" % r + "%" for r in annualized_returns
    ]

    return (
        annualized_volatility,
        annualized_returns,
        annualized_volatility_2dp,
        annualized_returns_2dp,
    )


def get_top_and_bottom_three(correlation_matrix):
    """
    get a list of the top 3 and bottom 3 most/least correlated assests
    for each node.

    Args:
        df (pd.DataFrame): pandas correlation matrix

    Returns:
        top_3_list (list): list of lists containing the top 3 correlations
            (name and value)
        bottom_3_list (list): list of lists containing the bottom three
            correlations (name and value)
    """
    df = correlation_matrix
    top_3_list = []
    bottom_3_list = []

    for col in df.columns:

        # exclude self correlation #reverse order of the list returned
        top_3 = list(np.argsort(abs(df[col]))[-4:-1][::-1])
        # bottom 3 list is returned in correct order
        bottom_3 = list(np.argsort(abs(df[col]))[:3])

        # get column index
        col_index = df.columns.get_loc(col)

        # find values based on index locations
        top_3_values = [df.index[x] + ": %.2f" % df.iloc[x, col_index] for x in top_3]
        bottom_3_values = [
            df.index[x] + ": %.2f" % df.iloc[x, col_index] for x in bottom_3
        ]

        top_3_list.append(convert_rankings_to_string(top_3_values))
        bottom_3_list.append(convert_rankings_to_string(bottom_3_values))

    return top_3_list, bottom_3_list


def get_coordinates(mst):
    """Returns the positions of nodes and edges in a format
    for Plotly to draw the network
    """
    # get list of node positions
    pos = nx.fruchterman_reingold_layout(mst)

    Xnodes = [pos[n][0] for n in mst.nodes()]
    Ynodes = [pos[n][1] for n in mst.nodes()]

    Xedges = []
    Yedges = []
    for e in mst.edges():
        # x coordinates of the nodes defining the edge e
        Xedges.extend([pos[e[0]][0], pos[e[1]][0], None])
        Yedges.extend([pos[e[0]][1], pos[e[1]][1], None])

    return Xnodes, Ynodes, Xedges, Yedges


def pipeline_graph(df,threshold,pct_change,days):
    raw_df = df.pct_change(pct_change).tail(days)

    # calculate correlation matrix using inbuilt pandas function
    correlation_matrix = raw_df.corr()
    correlation_matrix.index.name= 'index'


    # convert matrix to list of edges and rename the columns
    edges = correlation_matrix.stack().reset_index()


    edges.columns = ["asset_1", "asset_2", "correlation"]

    # remove self correlations
    edges = edges.loc[edges["asset_1"] != edges["asset_2"]].copy()

    # create undirected graph with weights corresponding to the correlation magnitude
    G0 = nx.from_pandas_edgelist(edges, "asset_1", "asset_2", edge_attr=["correlation"])

    # 'winner takes all' method - set minimum correlation threshold to remove some
    # edges from the diagram
    #threshold = threshold

    # create a new graph from edge list
    Gx = nx.from_pandas_edgelist(edges, "asset_1", "asset_2", edge_attr=["correlation"])

    # list to store edges to remove
    remove = []
    # loop through edges in Gx and find correlations which are below the threshold
    for asset_1, asset_2 in Gx.edges():
        corr = Gx[asset_1][asset_2]["correlation"]
        # add to remove node list if abs(corr) < threshold
        if abs(corr) < threshold:
            remove.append((asset_1, asset_2))

    # remove edges contained in the remove list
    Gx.remove_edges_from(remove)

    # assign colours to edges depending on positive or negative correlation
    # assign edge thickness depending on magnitude of correlation
    edge_colours = []
    edge_width = []
    for key, value in nx.get_edge_attributes(Gx, "correlation").items():
        edge_colours.append(assign_colour(value))
        edge_width.append(assign_thickness(value))

    # assign node size depending on number of connections (degree)
    node_size = []
    for key, value in dict(Gx.degree).items():
        node_size.append(assign_node_size(value))

    # create minimum spanning tree layout from Gx
    # (after small correlations have been removed)
    mst = nx.minimum_spanning_tree(Gx)

    edge_colours = []

    # assign edge colours
    for key, value in nx.get_edge_attributes(mst, "correlation").items():
        edge_colours.append(assign_colour(value))

    # get statistics for tooltip
    # make list of node labels.
    node_label = list(mst.nodes())
    # calculate annualised returns, annualised volatility and round to 2dp
    annual_vol, annual_ret, annual_vol_2dp, annual_ret_2dp = calculate_stats(raw_df)
    # get top and bottom 3 correlations for each node
    top_3_corrs, bottom_3_corrs = get_top_and_bottom_three(correlation_matrix)


    #create tooltip string by concatenating statistics
    description = [
        f"<b>{node}</b>"
        + "<br>"
        + annual_ret_2dp[index]
        + "<br>"
        + annual_vol_2dp[index]
        + "<br><br>Strongest correlations with: "
        + "<br>"
        + top_3_corrs[index]
        + "<br>Weakest correlations with: "
        "<br>" + bottom_3_corrs[index]
        for index, node in enumerate(node_label)
    ]

    # get coordinates for nodes and edges
    Xnodes, Ynodes, Xedges, Yedges = get_coordinates(mst)

    # assign node colour depending on positive or negative annualised returns
    node_colour = [assign_colour(i) for i in annual_ret]

    # assign node size based on annualised returns size (scaled by a factor)
    node_size = [abs(x) ** 0.5 * 5 for x in annual_ret]

    # Plot graph

    # edges
    tracer = go.Scatter(
        x=Xedges,
        y=Yedges,
        mode="lines",
        line=dict(color="#DCDCDC", width=1),
        hoverinfo="none",
        showlegend=False,
    )

    
 
    # nodes
    tracer_marker = go.Scatter(
        x=Xnodes,
        y=Ynodes,
        mode="markers+text",
        textposition="top center",
        marker=dict(size=node_size, line=dict(width=1), color=node_colour),
        hoverinfo="text",
        hovertext=description,
        text=node_label,
        textfont=dict(size=7),
        showlegend=False,
    )


    axis_style = dict(
        title="",
        titlefont=dict(size=20),
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        showticklabels=False,
    )


    layout = dict(
        title=f"Interactive minimum spanning tree <br> Last = {days}, Rolling {pct_change} Returns",
        width=800,
        height=800,
        autosize=False,
        showlegend=False,
        xaxis=axis_style,
        yaxis=axis_style,
        hovermode="closest",
        plot_bgcolor="#fff",
    )


    fig = go.Figure()
    fig.add_trace(tracer)
    fig.add_trace(tracer_marker)
    fig.update_layout(layout)

    fig.show()

    display(
        HTML(
            """
            <p>Node sizes are proportional to the size of
            annualised returns.<br>Node colours signify positive
            or negative returns since beginning of the timeframe.</p>
            """
        )
    )



def pipeline_graph_with_fixed_rolling_window(df, threshold, pct_change, window_size):
    time_series_central_nodes = []  # Store central nodes for each time period

    for i in range(window_size, len(df) + 1):
        # Calculate rolling correlations for the current time period
        raw_df = df.pct_change(pct_change).iloc[i - window_size:i]
        correlation_matrix = raw_df.corr()
        correlation_matrix.index.name = 'index'

            # convert matrix to list of edges and rename the columns
        edges = correlation_matrix.stack().reset_index()


        edges.columns = ["asset_1", "asset_2", "correlation"]

        # remove self correlations
        edges = edges.loc[edges["asset_1"] != edges["asset_2"]].copy()

        # create undirected graph with weights corresponding to the correlation magnitude
        G0 = nx.from_pandas_edgelist(edges, "asset_1", "asset_2", edge_attr=["correlation"])

        # 'winner takes all' method - set minimum correlation threshold to remove some
        # edges from the diagram
        #threshold = threshold

        # create a new graph from edge list
        Gx = nx.from_pandas_edgelist(edges, "asset_1", "asset_2", edge_attr=["correlation"])

        # list to store edges to remove
        remove = []
        # loop through edges in Gx and find correlations which are below the threshold
        for asset_1, asset_2 in Gx.edges():
            corr = Gx[asset_1][asset_2]["correlation"]
            # add to remove node list if abs(corr) < threshold
            if abs(corr) < threshold:
                remove.append((asset_1, asset_2))

        # remove edges contained in the remove list
        Gx.remove_edges_from(remove)

        # assign colours to edges depending on positive or negative correlation
        # assign edge thickness depending on magnitude of correlation
        edge_colours = []
        edge_width = []
        for key, value in nx.get_edge_attributes(Gx, "correlation").items():
            edge_colours.append(assign_colour(value))
            edge_width.append(assign_thickness(value))

        # assign node size depending on number of connections (degree)
        node_size = []
        for key, value in dict(Gx.degree).items():
            node_size.append(assign_node_size(value))

        # create minimum spanning tree layout from Gx
        # (after small correlations have been removed)
        mst = nx.minimum_spanning_tree(Gx)

        # Calculate centrality measures for the minimum spanning tree
        centrality = nx.degree_centrality(mst)

        # Sort nodes by centrality (descending) and select the top 3
        top_central_nodes = sorted(centrality, key=centrality.get, reverse=True)[:3]


        # Calculate degree centrality for only the top 3 central nodes
        degree_centrality_top_3 = {node: centrality[node] for node in top_central_nodes}


        # Append time index, top central nodes, and degree centrality values
        time_series_central_nodes.append((raw_df.last_valid_index(), degree_centrality_top_3))

    return time_series_central_nodes

