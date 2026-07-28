import pydot
from SolidityParser import ParseStream
import re


# def replace_fn_name_withContract(in_call, out_call):
#     in_call_ins = [ ]
#     out_call_ins = [ ]
#     #in_call = fn_calls["in_call"]
#     #out_call = fn_calls["out_call"]
#     for i in in_call:
#         if i['code'] != "" and i['code'] is not None:
#             code_with_contract_name = i['code'].replace(i['fn_name'], f"{i['contract']}.{i['fn_name']}", 1)
#         else:
#             code_with_contract_name =  f"{i['contract']}.{i['fn_name']}"
#         in_call_ins.append(code_with_contract_name)

#     for o in out_call:
#         if o['code'] != "" and o['code'] is not None:
#             code_with_contract_name = o['code'].replace(o['fn_name'], f"{o['contract']}.{o['fn_name']}", 1)
#         else:
#             code_with_contract_name =  f"{o['contract']}.{o['fn_name']}"
#         out_call_ins.append(code_with_contract_name)
#     return in_call, out_call


def parse_dot_file(dot_file):
    graphs = pydot.graph_from_dot_data(dot_file)
    if graphs is None:
        return None, None
    graph = graphs[0]  # Assume there is only one graph
    subgraphs = graph.get_subgraphs()
    nodes = graph.get_nodes()
    edges = graph.get_edges()

    subgraph_info = []
    edge_info = []

    for subgraph in subgraphs:
        sg_info = {
            "name": subgraph.get_name(),
            "label": subgraph.get_label(),
            "nodes": [],
            "edges": [],
        }

        for node in subgraph.get_nodes():
            node_attributes = node.get_attributes()
            sg_info["nodes"].append(
                {
                    "name": node.get_name(),
                    "label": node_attributes.get("label", None),
                    "color": node_attributes.get("color", None),
                }
            )

        subgraph_info.append(sg_info)

    for edge in edges:
        edge_attributes = edge.get_attributes()
        edge_info.append(
            {
                "source": edge.get_source(),
                "destination": edge.get_destination(),
                "label": edge_attributes.get("label", None),
                "color": edge_attributes.get("color", None),
            }
        )

    return subgraph_info, edge_info


def _get_calling_relationships(subgraphs, edges, function_name, contract_name=None):
    relationships = []
    for edge in edges:
        source = edge["source"].strip('"')
        destination = edge["destination"].strip('"')

        # Extracting contract and function names from the source and destination
        source_contract, _, source_function = source.partition(".")
        destination_contract, _, destination_function = destination.partition(".")

        if function_name in (source_function, destination_function):
            if (
                contract_name is None
                or contract_name == source_contract
                or contract_name == destination_contract
            ):
                relationships.append(
                    {
                        "source": source,
                        "destination": destination,
                        "color": edge["color"].strip('"'),
                    }
                )
    return relationships


def get_calling_relationships(dot_file, function_name, contract_name):
    subgraphs, edges = parse_dot_file(dot_file)
    if edges is None:
        return None
    relationships = _get_calling_relationships(
        subgraphs, edges, function_name, contract_name
    )

    # Output or process the relationships as needed
    # print("Calling Relationships:")
    # print(relationships)
    return relationships


def parse_calling_relationship(
    dot_file, function_name, contract_name, source_code, identifier
):
    call_relationships = get_calling_relationships(
        dot_file, function_name, contract_name
    )
    if call_relationships is None:
        return None, None
    parser = ParseStream(source_code)
    res = {"in_fn": [], "out_fn": []}
    for edge in call_relationships:
        if identifier == edge["source"]:
            store_dic = res["out_fn"]
            tart_fn = edge["destination"]
        else:
            store_dic = res["in_fn"]
            tart_fn = edge["source"]
        contract, fn = tart_fn.split(".")
        edge_color = edge["color"]
        if edge_color == "green":
            edge_type = "internal"
            fn_code = parser.get_function_context(contract, fn)
            if fn_code is None:
                fn_code = ""
        else:
            edge_type = "external"
            fn_code = ""
        store_dic.append(
            {"code": fn_code, "type": edge_type, "contract": contract, "fn_name": fn}
        )
    return res["in_fn"], res["out_fn"]


def parse_calling_relationship(call_relationships, parser, identifier):
    res = {"in_fn": [], "out_fn": []}
    print(identifier)
    for edge in call_relationships:
        if identifier == edge["source"]:
            store_dic = res["out_fn"]
            tart_fn = edge["destination"]
        else:
            store_dic = res["in_fn"]
            tart_fn = edge["source"]
        contract, fn = ".".join(tart_fn.split(".")[:-1]), tart_fn.split(".")[-1]
        edge_color = edge["color"]
        if edge_color == "green":
            edge_type = "internal"
            fn_code = parser.get_function_context(contract, fn)
            if fn_code is None:
                fn_code = ""
        else:
            edge_type = "external"
            fn_code = ""
        store_dic.append(
            {"code": fn_code, "type": edge_type, "contract": contract, "fn_name": fn}
        )
    return res["in_fn"], res["out_fn"]


def get_call_relationship_locations(locs_url, source_code, dot_call_graph):
    global count
    function_call_relationship = {}
    locs = re.findall(
        r"https://github.com/(.+?)/(?:blob|tree)/(.+?)/(.+?)#(.+)", locs_url
    )
    line_number = locs[0][-1]
    line_number = line_number.replace("L", "")
    line_number = line_number.replace("#", "")
    line_number = line_number.replace(":", "-")
    line_number = line_number.replace(")", "")
    line_number = line_number.replace("_", "-")
    try:
        if "-" in line_number:
            start, end = line_number.split("-")
            start = int(start) - 1
            end = int(end) - 1
            start = max(0, int(start))
            end = min(start, end)
        else:
            line_number = int(line_number)
            start = max(0, line_number - 1)
            end = start
    except:
        print(locs_url)

    parser = ParseStream(source_code)
    for i in range(start, end + 1):
        res = parser.get_function_from_loc(i)
        if res is not None:
            contract, function, context = res
            # function_code[hash(contract +"\n"+ function +"\n"+context)] = ( contract, function, context )
            call_relationships = get_calling_relationships(
                dot_call_graph, function, contract
            )
            # if call_relationships is None:
            function_call_relationship[
                hash(contract + "\n" + function + "\n" + context)
            ] = {"in_call": {}, "out_call": {}}
            #    continue
            in_calls, out_calls = parse_calling_relationship(
                call_relationships, parser, f"{contract}.{function}"
            )
            function_call_relationship[
                hash(contract + "\n" + function + "\n" + context)
            ] = {"in_call": in_calls, "out_call": out_calls}
        else:
            # the line code is out of the function
            # contract, function, context = parser.get_locs_code(i)
            continue
    return function_call_relationship


if __name__ == "__main__":
    main()
