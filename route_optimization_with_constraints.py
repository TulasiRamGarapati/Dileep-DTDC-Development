
import pandas as pd
from geopy.distance import geodesic
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Load dataset
df = pd.read_csv("sample_10000_Dataset.csv")

# Filter necessary columns and clean data
df_cleaned = df[[
    'order_id', 'customer_city', 'customer_state', 'seller_city', 'seller_state',
    'product_weight_g', 'freight_value'
]].dropna()

# Sample data
sample_df = df_cleaned.head(10)

# City coordinates (mocked for simplicity)
city_coords = {
    'sao paulo (SP)': (-23.5505, -46.6333),
    'barra (BA)': (-12.5833, -43.1667),
    'santo andre (SP)': (-23.6639, -46.5383),
    'belo horizonte (MG)': (-19.9167, -43.9345),
    'itatiba (SP)': (-23.0037, -46.8467),
    'juiz de fora (MG)': (-21.7642, -43.3503),
    'campo largo (PR)': (-25.4635, -49.5283),
    'salvador (BA)': (-12.9714, -38.5014),
    'toledo (PR)': (-24.7132, -53.7431),
    'rio de janeiro (RJ)': (-22.9068, -43.1729)
}

# Create data model
def create_data_model(sample_df, city_coords):
    locations = []
    demands = []
    time_windows = []
    order_ids = []

    for idx, row in sample_df.iterrows():
        seller = f"{row['seller_city']} ({row['seller_state']})"
        customer = f"{row['customer_city']} ({row['customer_state']})"
        
        if seller in city_coords and customer in city_coords:
            locations.append(city_coords[seller])
            demands.append(-int(row['product_weight_g']))
            time_windows.append((0, 240))
            order_ids.append(row['order_id'])

            locations.append(city_coords[customer])
            demands.append(int(row['product_weight_g']))
            time_windows.append((120, 480))
            order_ids.append(row['order_id'])

    return {
        'locations': locations,
        'demands': demands,
        'time_windows': time_windows,
        'num_vehicles': 2,
        'vehicle_capacities': [15000, 15000],
        'depot': 0,
        'order_ids': order_ids
    }

data = create_data_model(sample_df, city_coords)

# Create routing index manager
manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])

# Create Routing Model
routing = pywrapcp.RoutingModel(manager)

# Distance callback
def distance_callback(from_index, to_index):
    from_node = data['locations'][manager.IndexToNode(from_index)]
    to_node = data['locations'][manager.IndexToNode(to_index)]
    return int(geodesic(from_node, to_node).km)

transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Capacity constraint
def demand_callback(from_index):
    return data['demands'][manager.IndexToNode(from_index)]

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity'
)

# Time windows constraint
def time_callback(from_index, to_index):
    from_node = data['locations'][manager.IndexToNode(from_index)]
    to_node = data['locations'][manager.IndexToNode(to_index)]
    return int(geodesic(from_node, to_node).km * 2)  # 2 min/km

time_callback_index = routing.RegisterTransitCallback(time_callback)
routing.AddDimension(
    time_callback_index, 30, 480, False, 'Time'
)

time_dimension = routing.GetDimensionOrDie('Time')
for location_idx, time_window in enumerate(data['time_windows']):
    index = manager.NodeToIndex(location_idx)
    time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

# Solve
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
solution = routing.SolveWithParameters(search_parameters)

# Output solution
if solution:
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        print(f"\nRoute for vehicle {vehicle_id}:")
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            arrival = solution.Value(time_dimension.CumulVar(index))
            order = data['order_ids'][node_index] if node_index < len(data['order_ids']) else "Depot"
            print(f"  Location: {data['locations'][node_index]}, Order ID: {order}, Arrival Time: {arrival}")
            index = solution.Value(routing.NextVar(index))
        print("  Return to Depot")
else:
    print("No solution found.")
