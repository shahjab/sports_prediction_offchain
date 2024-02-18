from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType

client.team_box_scores(
    day=12, month=2, year=2024, 
    output_type=OutputType.JSON, 
    output_file_path="./12_2_2024_box_scores.json"
)