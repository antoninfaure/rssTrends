import { useEffect, useMemo, useRef, useState } from "react"
import * as d3 from "d3"

const NODE_RADIUS = 20
const FONT_MIN = 28
const FONT_MAX = 84

function buildFontScale(nodes) {
  if (!nodes?.length) {
    return () => FONT_MIN
  }
  const extent = d3.extent(nodes, (d) => d.size) || [1, 10]
  const min = extent[0] ?? 1
  const max = extent[1] ?? min + 1
  return d3.scaleSqrt().domain([Math.max(1, min), Math.max(1, max || min + 1)]).range([FONT_MIN, FONT_MAX]).clamp(true)
}

export function Graph({ nodes = [], links = [], title }) {
  const containerRef = useRef(null)
  const svgRef = useRef(null)
  const [dimensions, setDimensions] = useState({ width: 960, height: 600 })

  useEffect(() => {
    if (!containerRef.current) return
    const element = containerRef.current
    setDimensions({
      width: element.clientWidth || 960,
      height: Math.max(element.clientHeight || 0, 480),
    })
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        setDimensions({ width, height: Math.max(height, 480) })
      }
    })
    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const svgElement = svgRef.current
    if (!svgElement) return

    const width = dimensions.width || 960
    const height = dimensions.height || 600

    const svg = d3.select(svgElement)
    svg.selectAll("*").remove()
    svg.attr("viewBox", `0 0 ${width} ${height}`)

    const nodeData = nodes.map((node) => ({ ...node }))
    const linkData = links.map((link) => ({ ...link }))

    const color = d3
      .scaleLinear()
      .domain([1, d3.max(nodeData, (d) => d.size) || 1])
      .range(["#fde047", "#ef4444"])

    const fontScale = buildFontScale(nodeData)

    const zoomable = svg.append("g").attr("class", "zoomable")

    const zoomHandler = d3.zoom().scaleExtent([0.05, 2.5]).on("zoom", (event) => {
      zoomable.attr("transform", event.transform)
    })

    svg.call(zoomHandler)
    const initial = d3.zoomIdentity.translate(width / 2, height / 2).scale(0.35)
    svg.call(zoomHandler.transform, initial)

    const link = zoomable
      .append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(linkData)
      .enter()
      .append("line")
      .attr("stroke", "#1f2937")
      .attr("stroke-width", (d) => Math.max(1.5, d.size ? Math.sqrt(d.size) : 1.5))
      .attr("stroke-opacity", 0.4)

    const node = zoomable
      .append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(nodeData)
      .enter()
      .append("g")
      .attr("class", "node")

    node
      .append("circle")
      .attr("r", (d) => d.size * NODE_RADIUS)
      .attr("fill", (d) => color(d.size))
      .attr("fill-opacity", 0.9)
      .attr("stroke", "#0f172a")
      .attr("stroke-width", 1.5)

    node
      .append("text")
      .attr("dy", "0.35em")
      .attr("text-anchor", "middle")
      .text((d) => d.label)
      .style("font-family", "var(--font-sans)")
      .style("font-size", (d) => `${fontScale(d.size)}px`)
      .style("fill", "#020817")

    const simulation = d3
      .forceSimulation(nodeData)
      .force("link", d3.forceLink(linkData).id((d) => d.id))
      .force("charge", d3.forceManyBody().strength(-280))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide().radius((d) => d.size * NODE_RADIUS * 1.05).iterations(2))
      .on("tick", ticked)

    function ticked() {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y)

      node.attr("transform", (d) => `translate(${d.x},${d.y})`)
    }

    const drag = d3
      .drag()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart()
        d.fx = d.x
        d.fy = d.y
      })
      .on("drag", (event, d) => {
        d.fx = event.x
        d.fy = event.y
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0)
        d.fx = null
        d.fy = null
      })

    node.call(drag)

    if (title) {
      svg
        .append("text")
        .attr("class", "graph-title")
        .attr("x", width / 2)
        .attr("y", 40)
        .attr("text-anchor", "middle")
        .attr("fill", "#e2e8f0")
        .style("font-size", "1.5rem")
        .style("font-family", "var(--font-sans)")
        .text(title)
    }

    return () => {
      simulation.stop()
    }
  }, [nodes, links, title, dimensions])

  const memoizedTitle = useMemo(() => title, [title])

  return (
    <section className="flex h-full flex-col gap-6">
      <div ref={containerRef} className="relative h-[70vh] min-h-[26rem] w-full overflow-hidden rounded-3xl border border-border bg-card/70 backdrop-blur">
        <svg ref={svgRef} className="h-full w-full" role="img" aria-label={memoizedTitle ?? "Carte des tendances"} />
      </div>
    </section>
  )
}
