fetch("../data/list.json")
  .then(response => {
    return response.json();
  })
  .then(date_list => {
    let latest_date = date_list[0].name

    $('#dataInput').html('')
    date_list.forEach(x => {
      $('#dataInput').append(`<option value="${x.name}">${x.name}</option>`)
    })


    fetch(`../data/${latest_date}/edges.json`)
      .then(response => {
        if (response.status == 404) throw error;
        return response.json();
      })
      .then(links => {
        fetch(`../data/${latest_date}/nodes.json`)
          .then(response => {
            if (response.status == 404) throw error;
            return response.json();
          })
          .then(nodes => {
            const title = 'Actualités du 14-02-2023'
            const width = $('#mynetwork').innerWidth()
            const height = $('#mynetwork').innerHeight()



            var initial_zoom = d3.zoomIdentity.translate(600, 400).scale(0.05);

            //add zoom capabilities 
            var zoom_handler = d3.zoom().on("zoom", zoom_actions);

            const svg = d3.select('#mynetwork')
              .attr('width', width)
              .attr('height', height)
              .call(zoom_handler)
              .call(zoom_handler.transform, initial_zoom)

            /////// SLIDER ///////
            function addSlider() {
              var L = 10;
              var slider_size = 0.75 * width;
              var left_margin = 0.5 * (width - slider_size);

              var x = d3.scaleLinear()
                .domain([0, 10])
                .range([left_margin, slider_size + left_margin])
                .clamp(true);

              var slider = svg.append("g")
                .attr("transform", "translate(15," + (height - 50) + ")");

              slider.append("line")
                .attr("class", "track")
                .attr("x1", x.range()[0])
                .attr("x2", x.range()[1])
                .select(function () { return this.parentNode.appendChild(this.cloneNode(true)); })
                .attr("class", "track-inset")
                .select(function () { return this.parentNode.appendChild(this.cloneNode(true)); })
                .attr("class", "track-overlay")
                .call(d3.drag()
                  .on("start.interrupt", function () { slider.interrupt(); })
                  .on("start drag", function () { return hue(x.invert(d3.event.x)); }));

              var days = ['10-02-2023', '14-02-2023', '30-01-2023']
              var dx = L / (days.length - 1)
              var xticks = d3.range(0, L + dx, dx)

              slider.insert("g", ".track-overlay")
                .attr("class", "ticks")
                .attr("transform", "translate(0," + 25 + ")")
                .selectAll("text")
                .data(xticks)
                .enter().append("text")
                .attr("x", x)
                .attr("text-anchor", "middle")
                .text(function (d, i) { return days[i]; });

              var handle = slider.insert("circle", ".track-overlay")
                .attr("class", "handle")
                .attr("r", 9)
                .attr("cx", x.range()[0]); //initial position to zero

              function hue(h) {
                handle.attr("cx", x(h));
              }
            }
            var max_value = 0
            for (node of nodes) {
              if (node.size > max_value) max_value = node.size;
            }
            var color = d3.scaleLinear()
              .domain([1, max_value])
              .range(["yellow", "red"])

            const radius = 20

            var simulation = d3.forceSimulation()
              .force("link", d3.forceLink().id(function (d) { return d.id; }))
              .force("charge", d3.forceManyBody())
              .force("center", d3.forceCenter(width / 2, height / 2))
              .force("collide", d3.forceCollide().radius(d => { return (d.size * 3) * radius }).iterations(3))
              .on("tick", ticked);


            var zoomable = svg.append("g").attr("class", "zoomable").attr('transform', initial_zoom),
              link = zoomable.append("g").attr('class', 'links').selectAll(".link"),
              node = zoomable.append("g").attr('class', 'nodes').selectAll(".node")


            // Create a drag handler and append it to the node object instead
            var drag_handler = d3.drag()
              .on("start", dragstarted)
              .on("drag", dragged)
              .on("end", dragended);


            restart()

            function addJSON(json_nodes, json_links) {
              for (let x = 0; x < nodes.length; x++) {
                let found = false
                for (let y = 0; y < json_nodes.length; y++) {
                  if (json_nodes[y].id == nodes[x].id) {
                    found = true
                  }
                }
                if (found == false) {
                  nodes.splice(x, 1)
                }
              }
              for (let x = 0; x < json_nodes.length; x++) {
                var found = false;
                for (let y = 0; y < nodes.length; y++) {
                  if (nodes[y].id == json_nodes[x].id) {
                    found = true;
                    nodes[y].size = json_nodes[x].size
                    let theNode = zoomable.selectAll(".node")
                      .filter(function (d) {
                        return d.id == nodes[y].id;
                      })
                    theNode.select('circle')
                      .attr('r', function (d) {
                        return nodes[y].size * radius
                      })
                      .attr("fill", function (d) {
                        return color(nodes[y].size);
                      })
                    theNode.select('text')
                      .style("font-size", function (d) {
                        return nodes[y].size * radius
                      })

                  }
                }
                if (found == false) {
                  nodes.push(json_nodes[x]);
                }
              }
              let kept_nodes = nodes.map(x => x.id)
              let kept_links = []
              for (let i = 0; i < links.length; i++) {
                if (kept_nodes.includes(links[i].source.id) && kept_nodes.includes(links[i].target.id)) {
                  kept_links.push(links[i])
                }
              }
              /*for (let i=0; i < json_links.length; i++) {
                if (kept_nodes.includes(json_links[i].source.id) && kept_nodes.includes(json_links[i].target.id)) {
                  kept_links.push(json_links[i])
                }
              }*/

              links = kept_links.concat(json_links)

              restart();
            }

            $('#dataInput').on('change', function (event) {
              //if (event.keyCode == 13) {
              var optionSelected = $("option:selected", this);
              var valueSelected = this.value;
              $('#dateAlert').html(``)
              if (valueSelected.match(/^(\d{1,2})-(\d{1,2})-(\d{4})$/)) {
                let date = valueSelected
                fetch(`../data/${date}/nodes.json`)
                  .then(response => {
                    if (response.status == 404) throw error;
                    return response.json();
                  })
                  .then(new_nodes => {
                    fetch(`../data/${date}/edges.json`)
                      .then(response => {
                        if (response.status == 404) throw error;
                        return response.json();
                      })
                      .then(new_edges => {
                        //addJSON(new_nodes, new_edges)
                        links = new_edges
                        nodes = new_nodes
                        svg.select('.title').text('Actualités du ' + date)
                        restart()
                      })
                      .catch(err => {
                        return $('#dateAlert').html(`Erreur. Pas de fichier "${date}/edges.json"`)
                      })
                  })
                  .catch(err => {
                    return $('#dateAlert').html(`Erreur. Pas de fichier "${date}/nodes.json"`)
                  })
              } else {
                $('#dateAlert').html(`Erreur. Mauvais format de date`)
              }
              //}
            })
            /*d3.timeout(function () {
              fetch("../static/assets/rss-trends/data/10-02-2023/nodes.json")
                .then(response => {
                  return response.json();
                })
                .then(new_nodes => {
                  fetch("../static/assets/rss-trends/data/10-02-2023/edges.json")
                    .then(response => {
                      return response.json();
                    })
                    .then(new_edges => {
                      addJSON(new_nodes, new_edges)
                    })
                })
            }, 3000)*/

            function restart() {
              node.remove()
              link.remove()

              link = zoomable.append("g").attr('class', 'links').selectAll(".link"),
                node = zoomable.append("g").attr('class', 'nodes').selectAll(".node")

              node = node.data(nodes, function (d) { return d.id }).call(function (a) {
                a.transition().attr("r", function (d) {
                  return d.size * radius
                })
                  .attr("fill", function (d) {
                    return color(d.size);
                  })
              })

              var selection = node.enter().append('g').attr('class', 'node')

              selection.append("circle")
                .call(function (node) {
                  node.transition().attr("r", function (d) {
                    return d.size * radius
                  })
                    .attr("fill", function (d) {
                      return color(d.size);
                    })
                })


              selection.append("text")
                .attr('class', 'text-label')
                .attr("text-anchor", "middle")
                .attr("dy", ".35em")
                .text(function (d) {
                  return d.label
                })
                .style("font-size", function (d) {
                  return d.size * radius
                })
                .style('fill', 'black')

              node = selection.merge(node)

              // Apply the general update pattern to the links.
              link = link.data(links, function (d) { return d.source.id + "-" + d.target.id; });
              link.exit().remove();
              link = link.enter().append("g").append("line")
                .call(function (link) {
                  link.transition()
                    .attr("stroke-opacity", 1)
                    .attr("stroke-width", function (d) { return 10 + 'px' })
                })
                .style('stroke', 'black').merge(link);

              drag_handler(node);

              simulation.nodes(nodes)

              simulation.force("link").links(links);

              simulation.alphaTarget(0.3).restart();
              d3.timeout(function () {
                simulation.alphaTarget(0);
              }, 500)


            }

            function ticked() {
              link
                .attr("x1", function (d) { return d.source.x; })
                .attr("y1", function (d) { return d.source.y; })
                .attr("x2", function (d) { return d.target.x; })
                .attr("y2", function (d) { return d.target.y; });

              node
                .attr("transform", function (d) {
                  return "translate(" + d.x + "," + d.y + ")";
                })
            }

            svg.append('g')
              .append('text')
              .attr('class', 'title')
              .attr('x', width / 2)
              .attr('y', 50)
              .attr('text-anchor', 'middle')
              .text(title);

            //addSlider()

            function dragstarted(d) {
              if (!d3.event.active) simulation.alphaTarget(0.3).restart();
              d.fx = d.x;
              d.fy = d.y;
            }

            function dragged(d) {
              d.fx = d3.event.x;
              d.fy = d3.event.y;
            }

            function dragended(d) {
              if (!d3.event.active) simulation.alphaTarget(0);
              d.fx = null;
              d.fy = null;
            }

            function zoom_actions() {
              if (zoomable) {
                zoomable.attr("transform", d3.event.transform)
              }
            }
          })
          .catch(err => {
            console.error("No data for latest date")
          })
      })
      .catch(err => {
        console.error("No data for latest date")
      })
  })
  
