// *****PLEASE ENTER YOUR DETAILS BELOW*****
// T6-rm-mongo.mongodb.js

// Student ID:
// Student Name:

// Comments for your marker:

// ===================================================================================
// DO NOT modify or remove any of the comments below (items marked with //)
// ===================================================================================

// Use (connect to) your database - you MUST update xyz001
// with your authcate username

use("xyz001");

// (b)
// PLEASE PLACE REQUIRED MONGODB COMMAND TO CREATE THE COLLECTION HERE
// YOU MAY PICK ANY COLLECTION NAME
// ENSURE that your query is formatted and has a semicolon
// (;) at the end of this answer

// Drop collection


// Create collection and insert documents

// T6-rm-mongo.mongodb.js

// Drop collection if it exists
db.teams.drop();

// Create collection and insert documents (replace ... with actual JSON output from 6a)
db.teams.insertMany([
    // Paste JSON documents generated from SQL Task 6(a) here
    // Example (truncated):
    {
        "_id": 1,
        "carn_name": "RM Spring Series Clayton 2024",
        "carn_date": "22-Sep-2024",
        "team_name": "Champions",
        "team_leader": {
            "name": "Rob De Costella",
            "phone": "0422888999",
            "email": "rob@gmail.com"
        },
        "team_no_of_members": 4,
        "team_members": [
            // ... team members ...
        ]
    }
    // ... more team documents ...
]);

// List all documents
db.teams.find({});




// List all documents you added



// (c)
// PLEASE PLACE REQUIRED MONGODB COMMAND/S FOR THIS PART HERE
// ENSURE that your query is formatted and has a semicolon
// (;) at the end of this answer

// T6-rm-mongo.mongodb.js (continued)

// List teams with members who competed in 5 Km Run or 10 Km Run
db.teams.find(
    {
        "team_members.event_type": { $in: ["5 Km Run", "10 Km Run"] }
    },
    {
        "carn_date": 1,
        "carn_name": 1,
        "team_name": 1,
        "team_leader.name": 1,
        "_id": 0
    }
);



// (d)
// PLEASE PLACE REQUIRED MONGODB COMMAND/S FOR THIS PART HERE
// ENSURE that your query is formatted and has a semicolon
// (;) at the end of this answer


// (i) Add new team

// T6-rm-mongo.mongodb.js (continued)

// (i) Add the new team "The Great Runners"
db.teams.insertOne(
    {
        "_id": 101, // Manually decided _id
        "carn_name": "RM Winter Series Caulfield 2025",
        "carn_date": "29-Jun-2025",
        "team_name": "The Great Runners",
        "team_leader": {
            "name": "Jackson Bull",
            "phone": "0422412524",
            "email": "jackson.bull@email.com" // Manually decided email
        },
        "team_no_of_members": 1,
        "team_members": [
            {
                "competitor_name": "Jackson Bull",
                "competitor_phone": "0422412524",
                "event_type": "5 Km Run",
                "entry_no": 1, // Manually decided entry_no
                "starttime": "08:45:00",
                "finishtime": "-",
                "elapsedtime": "-"
            }
        ]
    }
);

// Show details of "The Great Runners" after insertion
db.teams.find(
    {
        "team_name": "The Great Runners",
        "carn_name": "RM Winter Series Caulfield 2025"
    }
);

// (ii) Add Steve Bull as a new team member
db.teams.updateOne(
    {
        "team_name": "The Great Runners",
        "carn_name": "RM Winter Series Caulfield 2025"
    },
    {
        $inc: { "team_no_of_members": 1 }, // Increment member count
        $push: {
            "team_members": {
                "competitor_name": "Steve Bull",
                "competitor_phone": "0422251427",
                "event_type": "10 Km Run",
                "entry_no": 2, // Next entry number for this event within the team context
                "starttime": "08:30:00",
                "finishtime": "-",
                "elapsedtime": "-"
            }
        }
    }
);

// Show details of "The Great Runners" after adding Steve Bull
db.teams.find(
    {
        "team_name": "The Great Runners",
        "carn_name": "RM Winter Series Caulfield 2025"
    }
);



// Illustrate/confirm changes made





// (ii) Add new team member





// Illustrate/confirm changes made


