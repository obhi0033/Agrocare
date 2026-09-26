
def weather_risk(temp,humidity,rainfall):
    reasons=[]; score=0
    if humidity>=80: score+=2; reasons.append("High humidity may support fungal disease development")
    if 18<=temp<=30: score+=1; reasons.append("Temperature is within a common disease-favorable range")
    if rainfall>0: score+=2; reasons.append("Recent rainfall can increase leaf wetness")
    return ("Higher environmental disease pressure" if score>=4 else "Moderate environmental disease pressure" if score>=1 else "Lower environmental disease pressure", reasons)

def weather_risk_breakdown(temp,humidity,rainfall):
    """Return every rule that was checked (met or not), so the UI can show
    exactly why a given assessment was reached instead of only the triggered ones."""
    return [
        {"key":"humidity","met":humidity>=80,"detail":f"Humidity {humidity}% (trigger: >=80%)"},
        {"key":"temp","met":18<=temp<=30,"detail":f"Temperature {temp}\u00b0C (trigger: 18-30\u00b0C)"},
        {"key":"rainfall","met":rainfall>0,"detail":f"Rainfall {rainfall}mm (trigger: >0mm)"},
    ]
