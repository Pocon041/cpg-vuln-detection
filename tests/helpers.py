from __future__ import annotations

from pathlib import Path


def write_graphml(path: Path, *, include_macro_helper: bool = False) -> None:
    helper = """
    <node id="90">
      <data key="labelV">METHOD</data>
      <data key="node__METHOD__NAME">MACRO_HELPER</data>
      <data key="node__METHOD__SIGNATURE">void()</data>
      <data key="node__METHOD__IS_EXTERNAL">false</data>
      <data key="node__METHOD__CODE">void MACRO_HELPER()</data>
    </node>
    <node id="91">
      <data key="labelV">BLOCK</data>
      <data key="node__BLOCK__CODE">{}</data>
    </node>
    <edge source="90" target="91"><data key="labelE">AST</data></edge>
    """ if include_macro_helper else ""
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="labelV" for="node" attr.name="labelV" attr.type="string"/>
  <key id="labelE" for="edge" attr.name="labelE" attr.type="string"/>
  <key id="node__METHOD__NAME" for="node" attr.name="NAME" attr.type="string"/>
  <key id="node__METHOD__SIGNATURE" for="node" attr.name="SIGNATURE" attr.type="string"/>
  <key id="node__METHOD__IS_EXTERNAL" for="node" attr.name="IS_EXTERNAL" attr.type="boolean"/>
  <key id="node__METHOD__CODE" for="node" attr.name="CODE" attr.type="string"/>
  <key id="node__CALL__CODE" for="node" attr.name="CODE" attr.type="string"/>
  <key id="node__CALL__LINE_NUMBER" for="node" attr.name="LINE_NUMBER" attr.type="int"/>
  <key id="node__IDENTIFIER__CODE" for="node" attr.name="CODE" attr.type="string"/>
  <key id="node__IDENTIFIER__LINE_NUMBER" for="node" attr.name="LINE_NUMBER" attr.type="int"/>
  <key id="node__BLOCK__CODE" for="node" attr.name="CODE" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="1">
      <data key="labelV">METHOD</data>
      <data key="node__METHOD__NAME">target</data>
      <data key="node__METHOD__SIGNATURE">int(char*)</data>
      <data key="node__METHOD__IS_EXTERNAL">false</data>
      <data key="node__METHOD__CODE">int target(char *src)</data>
    </node>
    <node id="2"><data key="labelV">BLOCK</data><data key="node__BLOCK__CODE">{{...}}</data></node>
    <node id="3"><data key="labelV">CALL</data><data key="node__CALL__CODE">strcpy(dst, src)</data><data key="node__CALL__LINE_NUMBER">4</data></node>
    <node id="4"><data key="labelV">IDENTIFIER</data><data key="node__IDENTIFIER__CODE">src</data><data key="node__IDENTIFIER__LINE_NUMBER">4</data></node>
    <node id="5"><data key="labelV">METHOD</data><data key="node__METHOD__NAME">strcpy</data><data key="node__METHOD__IS_EXTERNAL">true</data></node>
    <node id="6"><data key="labelV">METHOD</data><data key="node__METHOD__NAME">&lt;global&gt;</data><data key="node__METHOD__IS_EXTERNAL">false</data></node>
    {helper}
    <edge source="1" target="2"><data key="labelE">AST</data></edge>
    <edge source="2" target="3"><data key="labelE">AST</data></edge>
    <edge source="3" target="4"><data key="labelE">AST</data></edge>
    <edge source="3" target="4"><data key="labelE">CFG</data></edge>
    <edge source="3" target="4"><data key="labelE">CDG</data></edge>
    <edge source="4" target="3"><data key="labelE">REACHING_DEF</data></edge>
    <edge source="3" target="5"><data key="labelE">CALL</data></edge>
  </graph>
</graphml>
""",
        encoding="utf-8",
    )

