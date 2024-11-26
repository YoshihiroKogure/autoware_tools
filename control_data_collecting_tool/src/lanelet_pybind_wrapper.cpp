#include <pybind11/pybind11.h>
#include <autoware_map_msgs/msg/lanelet_map_bin.hpp>
#include <lanelet2_core/Forward.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>
#include <lanelet2_routing/RoutingGraph.h>
#include <autoware_lanelet2_extension/utility/message_conversion.hpp>

namespace py = pybind11;

// Wrapper function for fromBinMsg that returns the map
lanelet::LaneletMapPtr fromBinMsgWrapper(const autoware_map_msgs::msg::LaneletMapBin& msg) {
    auto map = std::make_shared<lanelet::LaneletMap>();
    try {
        lanelet::utils::conversion::fromBinMsg(msg, map);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error in fromBinMsg: ") + e.what());
    }
    return map;
}

PYBIND11_MODULE(lanelet_pybind_module, m) {
    m.def("from_bin_msg", &fromBinMsgWrapper,
          py::arg("msg"),
          "Converts LaneletMapBin message to a LaneletMap object and returns it.");
}
