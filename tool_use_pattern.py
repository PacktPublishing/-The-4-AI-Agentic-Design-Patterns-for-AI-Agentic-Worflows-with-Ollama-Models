import os
from typing import List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from autogen import ConversableAgent
from dotenv import load_dotenv

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
# llm_config = {"model": model, "temperature": 0.0, "api_key": api_key}

# == Using the local Ollama model ==
llm_config = {"config_list": config_list, "temperature": 0.0}


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
    except:
        return False


class TravelAgentSystem:
    def __init__(self, llm_config: dict):
        self.tools = TravelTools()

        agents = {
            "travel_assistant": (
                "TravelAssistant",
                "You are a helpful AI travel assistant. Add 'TERMINATE' when task is complete.",
            ),
            "planning_agent": (
                "PlanningAgent",
                "You create optimal travel itineraries. Add 'TERMINATE' when planning is done.",
            ),
            "booking_agent": (
                "BookingAgent",
                "You handle booking queries. Add 'TERMINATE' when booking info is provided.",
            ),
        }

        for attr, (name, sys_msg) in agents.items():
            setattr(
                self,
                attr,
                ConversableAgent(
                    name=name, system_message=sys_msg, llm_config=llm_config
                ),
            )

        self.user_proxy = ConversableAgent(
            name="UserProxy",
            is_termination_msg=check_termination,
            human_input_mode="NEVER",
        )

        self._register_tools()

    def _register_tools(self):
        tools = [
            self.tools.get_flight_status,
            self.tools.track_flight_prices,
            self.tools.get_hotel_details,
            self.tools.get_location_info,
        ]

        for tool in tools:
            self.travel_assistant.register_for_llm(
                name=tool.__name__,
                description=tool.__doc__ or f"Execute {tool.__name__}",
            )(tool)
            self.user_proxy.register_for_execution(name=tool.__name__)(tool)

    def initiate_conversation(self, message: str):
        return self.user_proxy.initiate_chat(self.travel_assistant, message=message)


if __name__ == "__main__":
    # llm_config = {
    #     "model": "gpt-4o-mini",
    #     "temperature": 0.0,
    #     "api_key": os.environ["OPENAI_API_KEY"],
    # }

    travel_system = TravelAgentSystem(llm_config)
    travel_system.initiate_conversation(
        "I need help planning a trip to New York next week. "
        "I need flight and hotel information, and what's going on in the city."
    )
