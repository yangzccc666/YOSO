const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const path=require('node:path')
const vm=require('node:vm')
const ts=require('typescript')
const context={exports:{},Date}
const source=fs.readFileSync(path.join(__dirname,'../src/workClock.ts'),'utf8')
vm.runInNewContext(ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,context)
const {DEFAULT_WORK_CLOCK:defaults,workShiftState,weeklyRestState,isWorkingDay,formatClockSeconds,upcomingFestivals,upcomingHolidays,daysBetween}=context.exports
const calendar={years:[2026,2027],sources:[],holidays:[
 {name:'中秋节',start:'2026-09-25',end:'2026-09-27'},
 {name:'元旦',start:'2027-01-01',end:'2027-01-03'},
 {name:'春节',start:'2027-02-05',end:'2027-02-13'}
],workdays:['2026-09-20','2027-02-14']}
const date=(month,day,hour=0,minute=0,second=0,year=2026)=>new Date(year,month-1,day,hour,minute,second)

test('work clock computes seconds and freezes at shift end without negative values',()=>{
 const active=workShiftState(date(11,6,15,20,30),defaults,calendar)
 assert.equal(active.status,'working');assert.equal(active.elapsedSeconds,22830);assert.equal(active.remainingSeconds,9570)
 const ended=workShiftState(date(11,6,23),defaults,calendar)
 assert.equal(ended.status,'finished');assert.equal(ended.elapsedSeconds,32400);assert.equal(ended.remainingSeconds,0)
 const before=workShiftState(date(11,6,8),defaults,calendar)
 assert.equal(before.status,'before');assert.equal(before.elapsedSeconds,0);assert.equal(before.remainingSeconds,36000)
})
test('night shift follows its starting workday into the weekend',()=>{
 const settings={...defaults,startTime:'22:00',endTime:'06:00'}
 const active=workShiftState(date(11,7,2),settings,calendar)
 assert.equal(active.status,'working');assert.equal(active.elapsedSeconds,14400);assert.equal(active.remainingSeconds,14400)
 assert.equal(active.start.getDate(),6);assert.equal(active.end.getDate(),7)
 assert.equal(weeklyRestState(date(11,7,2),settings,calendar).seconds,14400)
 assert.equal(workShiftState(date(11,7,6),settings,calendar).status,'rest')
 assert.equal(weeklyRestState(date(11,7,6),settings,calendar).ongoing,true)
})
test('double and single weekends begin at the last preceding shift end',()=>{
 assert.equal(weeklyRestState(date(11,6,17),defaults,calendar).seconds,3600)
 assert.equal(weeklyRestState(date(11,6,18),defaults,calendar).ongoing,true)
 assert.equal(weeklyRestState(date(11,9,9),defaults,calendar).ongoing,false)
 const single={...defaults,weekendMode:'single'}
 assert.equal(weeklyRestState(date(11,7,12),single,calendar).seconds,21600)
 assert.equal(weeklyRestState(date(11,7,12),defaults,calendar).ongoing,true)
 const wednesday={...single,singleRestDay:3}
 assert.equal(weeklyRestState(date(11,10,17),wednesday,calendar).seconds,3600)
})
test('official compensatory workdays override weekly rests and can be disabled',()=>{
 const sunday=date(9,20,10)
 assert.equal(isWorkingDay(sunday,defaults,calendar),true)
 assert.equal(workShiftState(sunday,defaults,calendar).makeup,true)
 assert.equal(weeklyRestState(sunday,defaults,calendar).ongoing,false)
 assert.equal(isWorkingDay(sunday,{...defaults,followChinaCalendar:false},calendar),false)
 assert.equal(isWorkingDay(date(9,25,10),defaults,calendar),false)
 assert.equal(isWorkingDay(date(9,25,10),{...defaults,followChinaCalendar:false},calendar),true)
})
test('holiday lists span years, include active ranges, and distinguish observances',()=>{
 assert.equal(upcomingHolidays(date(12,31),calendar)[0].start,'2027-01-01')
 assert.equal(upcomingHolidays(date(1,3,23,59,0,2027),calendar)[0].name,'元旦')
 assert.equal(upcomingHolidays(date(1,4,0,0,0,2027),calendar)[0].name,'春节')
 const festivals=upcomingFestivals(date(12,1),calendar)
 assert.equal(festivals.find(item=>item.name==='元旦').dayOff,true)
 assert.equal(festivals.find(item=>item.name==='国家宪法日').dayOff,false)
 const rest=workShiftState(date(2,6,10,0,0,2027),defaults,calendar)
 assert.equal(rest.status,'rest');assert.equal(rest.nextStart.getDate(),14)
 assert.equal(upcomingHolidays(date(12,1,0,0,0,2028),calendar).length,0)
})
test('calendar-day counts ignore time of day while clock formatting preserves seconds',()=>{
 assert.equal(daysBetween(date(12,31,23,59),date(1,1,0,1,0,2027)),1)
 assert.equal(formatClockSeconds(9570),'02小时 39分 30秒')
 assert.equal(formatClockSeconds(90061,true),'1天 01小时 01分 01秒')
 assert.equal(formatClockSeconds(-2),'00小时 00分 00秒')
})
test('festival dates show weekdays including next-year Valentine Sunday without making it a public holiday',()=>{
 const {formatCalendarDateRange}=context.exports
 assert.equal(formatCalendarDateRange('2027-02-14'),'2027-02-14（周日）')
 const valentine=upcomingFestivals(date(2,1,0,0,0,2027),calendar).find(item=>item.name==='情人节')
 assert.equal(valentine.dayOff,false)
 assert.equal(formatCalendarDateRange(valentine.start,valentine.end),'2027-02-14（周日）')
})
test('multi-day holiday ranges show both weekdays while single-day dates are not duplicated',()=>{
 const {formatCalendarDateRange}=context.exports
 assert.equal(formatCalendarDateRange('2027-01-01','2027-01-03'),'2027-01-01（周五） 至 2027-01-03（周日）')
 assert.equal(formatCalendarDateRange('2026-12-31','2027-01-01'),'2026-12-31（周四） 至 2027-01-01（周五）')
 assert.equal(formatCalendarDateRange('2028-02-29','2028-02-29'),'2028-02-29（周二）')
})
