--****PLEASE ENTER YOUR DETAILS BELOW****
--T3-rm-dm.sql

--Student ID:
--Student Name:

/* Comments for your marker:




*/

--(a)
-- Drop sequence for COMPETITOR if it exists
DROP SEQUENCE competitor_seq;

-- Create sequence for COMPETITOR
CREATE SEQUENCE competitor_seq
START WITH 100
INCREMENT BY 5;

-- Drop sequence for TEAM if it exists
DROP SEQUENCE team_seq;

-- Create sequence for TEAM
CREATE SEQUENCE team_seq
START WITH 100
INCREMENT BY 5;

--(b)
SET TRANSACTION NAME 'Register Keith and Jackson, form Super Runners';

-- Find the carnival date for 'RM Winter Series Caulfield 2025'
-- Find the event_id for '10 Km Run' in that carnival
-- Find char_id for 'Salvation Army' and 'RSPCA'

-- Insert Keith Rose
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone)
VALUES (competitor_seq.NEXTVAL, 'Keith', 'Rose', 'M', TO_DATE('15-JAN-1990', 'DD-MON-YYYY'), 'keith.rose@monash.edu', 'Y', '0422141112');

-- Insert Jackson Bull
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone)
VALUES (competitor_seq.NEXTVAL, 'Jackson', 'Bull', 'M', TO_DATE('20-MAR-1992', 'DD-MON-YYYY'), 'jackson.bull@monash.edu', 'Y', '0422412524');

-- Retrieve competitor numbers
-- Retrieve carnival date and event_id
-- Retrieve charity IDs

-- Insert team (assuming Keith's entry leads the team)
INSERT INTO TEAM (team_id, team_name, carn_date, event_id, entry_no)
VALUES (team_seq.NEXTVAL, 'Super Runners', (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025'),
        (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
         WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')
           AND et.eventtype_desc = '10 Km Run'),
        -- Determine entry_no for Keith's entry, assuming it's the first for this team leader for this event.
        -- This is a tricky part as entry_no is per event and reused.
        -- If rm-schema-insert.sql only creates event and carnival, and not any entries, then we can start from 1.
        1 -- This needs careful consideration based on Task 2 data.
       );

-- Insert Keith's entry
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id)
VALUES ((SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
         WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')
           AND et.eventtype_desc = '10 Km Run'),
        1, -- This corresponds to the entry_no in the TEAM table for the team leader
        TO_DATE('08:30:00', 'hh24:mi:ss'), NULL, NULL,
        (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422141112'),
        (SELECT team_id FROM TEAM WHERE team_name = 'Super Runners' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')),
        (SELECT char_id FROM CHARITY WHERE char_name = 'Salvation Army')
       );

-- Insert Jackson's entry
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id)
VALUES ((SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
         WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')
           AND et.eventtype_desc = '10 Km Run'),
        2, -- This is the next entry_no for the same event
        TO_DATE('08:30:05', 'hh24:mi:ss'), NULL, NULL,
        (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422412524'),
        (SELECT team_id FROM TEAM WHERE team_name = 'Super Runners' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')),
        (SELECT char_id FROM CHARITY WHERE char_name = 'RSPCA')
       );

COMMIT;


--(c)
SET TRANSACTION NAME 'Update Jackson Bull''s entry';

-- Retrieve Jackson's competitor number
-- Retrieve the carnival date for 'RM Winter Series Caulfield 2025'
-- Retrieve the event_id for the '10 Km Run' in that carnival (old event)
-- Retrieve the event_id for the '5 Km Run' in that carnival (new event)
-- Retrieve the char_id for 'Beyond Blue'

-- Delete Jackson's old entry for the 10 Km Run
DELETE FROM ENTRY
WHERE comp_no = (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422412524')
  AND event_id = (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
                  WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')
                    AND et.eventtype_desc = '10 Km Run');

-- Insert Jackson's new entry for the 5 Km Run
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id)
VALUES ((SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
         WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')
           AND et.eventtype_desc = '5 Km Run'),
        -- Determine entry_no for Jackson's new 5K entry.
        -- If this event has no entries yet, start from 1. Otherwise, find MAX(entry_no) + 1.
        -- For this task, we assume there are no other entries for this event yet, or we assign an appropriate new entry_no.
        (SELECT NVL(MAX(entry_no), 0) + 1 FROM ENTRY WHERE event_id = (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
                                                                       WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')
                                                                         AND et.eventtype_desc = '5 Km Run')),
        TO_DATE('08:45:00', 'hh24:mi:ss'), NULL, NULL, -- sensible start time for 5K
        (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422412524'),
        NULL, -- Jackson is now an individual runner as team is disbanded later in (d)
        (SELECT char_id FROM CHARITY WHERE char_name = 'Beyond Blue')
       );
COMMIT;

--(d)
SET TRANSACTION NAME 'Keith withdraws, Super Runners disband';

-- Retrieve Keith's comp_no
-- Retrieve Jackson's comp_no
-- Retrieve the carnival date for 'RM Winter Series Caulfield 2025'
-- Retrieve the team_id for 'Super Runners' for that carnival

-- If not already done in (c), update Jackson's entry to remove him from the team.
-- This assumes Jackson's entry was previously updated to the 5K run.
UPDATE ENTRY
SET team_id = NULL
WHERE comp_no = (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422412524')
  AND event_id IN (SELECT event_id FROM EVENT WHERE carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025'));

-- Delete Keith's entry for the RM Winter Series Caulfield 2025 carnival
DELETE FROM ENTRY
WHERE comp_no = (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422141112')
  AND event_id IN (SELECT event_id FROM EVENT WHERE carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025'));

-- Delete the Super Runners team for the RM Winter Series Caulfield 2025 carnival
DELETE FROM TEAM
WHERE team_name = 'Super Runners'
  AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025');

COMMIT;