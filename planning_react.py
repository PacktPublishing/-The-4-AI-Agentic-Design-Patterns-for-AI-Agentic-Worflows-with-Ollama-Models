from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from typing import List, Optional
from dotenv import load_dotenv
from autogen import (
    ConversableAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
)

load_dotenv()

model = "gpt-4o-mini"
api_key = os.environ["OPENAI_API_KEY"]

config_list = [
    {
        "model": "llama3.2:3b",  # add your own model here
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
]

# == Uncomment the following line to use OpenAI API ==
llm_config = {"model": model, "temperature": 0.0, "api_key": api_key}

# == Using the local Ollama model ==
# llm_config = {"config_list": config_list, "temperature": 0.0}


class ReActAgent(ConversableAgent):
    def __init__(self, name: str, system_message: str, llm_config: dict):
        react_prompt = """You are an AI agent that follows the ReAct pattern strictly:
THOUGHT: Reason clearly about the current situation and needs
ACTION: Select a specific action from available tools, providing required parameters
OBSERVATION: Analyze the results from the action
Reason about next steps based on all observations

Always format your responses exactly as:
THOUGHT: [reasoning about what to do next]
ACTION: [tool_name] {parameters}
OBSERVATION: [analysis of results]

Continue this cycle until the task is complete, then end with:
Thought: here's the summary... Task is complete. .
Action: TERMINATE
"""
        super().__init__(
            name=name,
            system_message=system_message + react_prompt,
            llm_config=llm_config,
        )


@dataclass
class FlightDetails:
    flight_number: str
    status: str
    departure: datetime
    arrival: datetime
    price: float
    seats_available: int

    def to_dict(self):
        return {
            "flight_number": self.flight_number,
            "status": self.status,
            "departure": self.departure.isoformat(),
            "arrival": self.arrival.isoformat(),
            "price": self.price,
            "seats_available": self.seats_available,
        }


@dataclass
class HotelDetails:
    name: str
    location: str
    price: float
    rating: float
    reviews: List[str]
    available_rooms: int

    def to_dict(self):
        return {
            "name": self.name,
            "location": self.location,
            "price": self.price,
            "rating": self.rating,
            "reviews": self.reviews,
            "available_rooms": self.available_rooms,
        }


@dataclass
class LocationInfo:
    weather: str
    events: List[str]
    safety_alerts: List[str]
    local_time: datetime

    def to_dict(self):
        return {
            "weather": self.weather,
            "events": self.events,
            "safety_alerts": self.safety_alerts,
            "local_time": self.local_time.isoformat(),
        }


class TravelTools:
    @staticmethod
    def get_flight_status(flight_number: str, date: Optional[str] = None) -> dict:
        return FlightDetails(
            flight_number=flight_number,
            status="On Time",
            departure=datetime.now(),
            arrival=datetime.now() + timedelta(hours=2),
            price=299.99,
            seats_available=15,
        ).to_dict()

    @staticmethod
    def track_flight_prices(origin: str, destination: str, date_range: str) -> dict:
        return {
            "price_history": [320.0, 310.0, 299.99],
            "price_forecast": [305.0, 315.0, 325.0],
        }

    @staticmethod
    def get_hotel_details(location: str, check_in: str, check_out: str) -> dict:
        return HotelDetails(
            name="Grand Hotel",
            location=location,
            price=199.99,
            rating=4.5,
            reviews=["Great location", "Excellent service"],
            available_rooms=5,
        ).to_dict()

    @staticmethod
    def get_location_info(location: str, date: Optional[str] = None) -> dict:
        return LocationInfo(
            weather="Sunny, 75°F",
            events=["Local Festival", "Art Exhibition"],
            safety_alerts=["No current alerts"],
            local_time=datetime.now(),
        ).to_dict()


def check_termination(msg):
    try:
        content = msg.get("content", "")
        if isinstance(content, str):
            if "TERMINATE" in content or any(
                term in content.lower()
                for term in ["completed", "here are the results", "finished"]
            ):
                return True
        return False
    except Exception:
        return False


class TravelAgentSystem:
    def __init__(self, llm_config: dict):

        self.tools = TravelTools()

        self.travel_assistant = ReActAgent(
            name="TravelAssistant",
            system_message="You plan travel using a systematic approach.",
            llm_config=llm_config,
        )

        self.user_proxy = ConversableAgent(
            name="UserProxy",
            is_termination_msg=check_termination,
            human_input_mode="NEVER",
        )

        self._register_tools()

    def _register_tools(self):
        tools = [
            (self.tools.get_flight_status, "Get current Flight Status"),
            (self.tools.track_flight_prices, "Track Flight Prices"),
            (self.tools.get_hotel_details, "Get Hotel Details"),
            (self.tools.get_location_info, "Get Location Info"),
        ]

        for tool, description in tools:
            # Register the function with its description for travel_assistant
            self.travel_assistant.register_for_llm(
                name=tool.__name__,
                description=description,
            )(tool)

            self.user_proxy.register_for_execution(name=tool.__name__)(tool)

    def run_query(self, query: str):
        return self.user_proxy.initiate_chat(self.travel_assistant, message=query)


def main():
    load_dotenv()
    travel_system = TravelAgentSystem(llm_config)

    # Test case demonstrating the ReAct pattern
    queries = [
        "Plan a trip to NYC: need flight AA123 status, hotel for next week, and local events",
        "Find the cheapes time to fly from SFO to NYC next month and suggest a hotels",
    ]

    for query in queries:
        print(f"\nUser Query: {query}")
        result = travel_system.run_query(query)
        print("-" * 50)
        # print(f"Agent Response: \n\n{result}")


if __name__ == "__main__":
    main()
