<?php
  header('Content-Type: text/html; charset=euc-kr');
?>
<?php session_start();
DEFINE('DB_USERNAME', 'root');
DEFINE('DB_PASSWORD', '');
DEFINE('DB_HOST', 'localhost');
DEFINE('DB_DATABASE', 'LearningEyes');

$mysqli = new mysqli(DB_HOST, DB_USERNAME, DB_PASSWORD, DB_DATABASE);

if (mysqli_connect_error()) {
    die('Connect Error ('.mysqli_connect_errno().') '.mysqli_connect_error());
}
?>
<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
  <title> CCTV CENTER </title>
  <link rel="stylesheet"
  href="http://code.jquery.com/ui/1.10.4/themes/redmond/jquery-ui.min.css"/>
  <style type="text/css">
  header{
    height: 40px;
    background-color: #19b092;
    margin: 0px;
  }
  section{
    height: 250px;
    background-color: #949494;
    margin: 5px;
  }
  body{
    background-color: #949494;
  }
  footer{
    height: 250px;
    background-color:#949494;
  }

#wrapper{
    display: flex;
    flex:1;
    margin-top: 7px;
    margin-left: 10px;
}

#side_chart{
  width:600px;
  height:525px;
  overflow:scroll;
  background-color:#828282;
}
  #side_right_up{
    
    margin-top: 5px;
    margin-left: 3px;
}
#side_left_bottom{
width:230px;
height:240px;
background-color:#828282;
float:left;
margin: 5px;
}
#container{
margin: 5px auto;
width: 350px;
height: 260px;
border: 5px #333 solid;
background-color: #666;
}
#CCTV_list{
  margin-left: 10px;
  margin-top: 35px;
}
#videoElement{
width: 300px;
margin-top: 30px;
margin-left: 40px;
}
#board_section{
 overflow:scroll;
    width: 400px;
    height: 268px;
    background-color:#828282;
    margin-left: 10px;
}
#side_right_bottom{

}

#search_section{
  display: block;
  margin-left: -1345px;
  margin-top: 280px;
  width:733px;
  height:245px;
  background-color:#828282;
overflow:scroll;
}
#side_right_bottom2{
margin-left: 5px;
margin-top: 5px;
}
#response{
  column-count:2;
}
ul{
  float: right;
}
td{
  font-size:9pt;
}
A:link{
  font:9pt;
  color black;
  text-decoration:none;
  font-familty:굴림;
  font-size:9pt;
}
A:visited{
  text-decoration:non;
  color:black;
  font-size:9pt;
}
  </style>

</head>

<!--방송/전화/문자 서비스 선택 알림창-->
<!--
<script type="text/javascript">
function goCheck(){
  var s="/";
  var chkBox=document.getElementsByName("agree");
  for (var i=0;i<chkBox.length;i++){
    if(chkBox[i].checked){
      s+=chkBox[i].value;
      s+="/";
    }
  }
  if(s!="/"){
    alert("Start "+s);
  }
  else{
    alert("Please select a service");
  }
}
</script>
-->
<body>
  <header id="head_alarm">
    <ul>
      <form method="POST" action="logout.php">
      <input type="submit" value="logout">&nbsp&nbsp
    </form>
  </ul>
  </header>

  <!--웹캠 임의로 연결해둔 부분 (나중에는 CCTV와 연결시켜야할 부분)-->
  <section id="wrapper">
    <div id="container">
      <div id="CCTV_list">
        <p style="color: white; text-align: center">CCTV LIST</p>
        <table border="1" style="background-color: white; text-align: center;border-color: #19b092">
          <th width="100">CCTV No</th>
          <th width="100">Screen No</th>
          <th width="100">Screen Place</th>
          <tr width="100"><!-- 첫번째 줄 시작 -->
            <td>1</td>
            <td>21</td>
            <td>Main Hall</td>
          </tr><!-- 첫번째 줄 끝 -->
        </table>
      </div>
      <div id="videoElement" >
        <p style="color: white; display: inline-block;font-size: 110%">CCTV NO : </p>
        <input type="text" name="camera_no" style="line-height: 30px; display: inline-block;" size="10">
        <button style="line-height: 30px; display: inline-block;" onclick="showPopup();" >ENTER</button> 
      </div>   
  </div>
  <script type="text/javascript">
    function showPopup() { window.open("http://localhost:5000", "a", "width=600, height=650, left=100, top=50"); }
  </script>
<!--    <script type="text/javascript">
    var video=document.querySelector("#videoElement");
    var vgaContraints={
      video:{
        mandatory:{
          minWidth: 1200,
          minHeight: 250
        }
      }
    };
    navigator.getUserMedia=navigator.getUserMedia||navigator.webkitGetUsermedia||
    navigator.mozGetUserMedia||navigator.msGetUserMedia||navigator.oGetUserMedia;
    if(navigator.getUserMedia){
      navigator.getUserMedia(vgaContraints,handleVideo,videoError);
    }
    function handleVideo(stream){
      video.srcObject=stream;
    }
    function videoError(e){
      alert("Check camera connection");
    }
  </script>
-->

<!-- 일지 -->
<section id="board_section">
  <div id="side_right_bottom">
    
  <table class="list-table">
    <thead>
      <tr>
        <th width="10"> No</th>
        <th width="500">Title</th>
        <th width="50"> Writer</th>
        <th width="300">Date</th>
        <th width="10"> View</th>
      </tr>
    </thead>
    <?php
    //$sql="select * from board order by wdate desc limit 0,10";
    $sql="select * from board order by wdate desc";
    $result=mysqli_query($mysqli,$sql);
    while($board=mysqli_fetch_assoc($result)){
      $title=$board['title'];
      if(strlen($title)>30){
        $title=str_replace($board['title'],mb_substr($board['title'],0,30,"utf-8")."...",$board['title']);
      }
      ?>
      <tbody>
        <tr>
          <td width="70" align=center><?php echo $board['id'];?></td>
          <td width="500" align=center><a href="read.php?id=<?php echo $board['id'];?>"><?php echo $title;?></a></td>
          <td width="120" align=center><?php echo $board['userid'];?></td>
          <td width="100" align=center><?php echo $board['wdate'];?></td>
          <td width="100" align=center><?php echo $board['view'];?></td>
        </tr>
      </tbody>
    <?php } ?>
  </table>
  <center>
  <form method="POST" action="write.php">
  <input type="submit" value="write">
</center>
</form>
  </div>
</section>


<section id="side_chart">
  <div id="side_right_up">
    CHART:
    <select name='chart' style="width:40px;">
      <option value='' selected>--option--</option>
      <option value='donut'>DONUT</option>
      <option value='bar'>BAR</option>
      <option value='line'>LINE</option>
    </select>
    X-axis:
    <select name='x' style="width:40px;">
      <option value='' selected>--option--</option>
      <option value='area'>AREA</option>
      <option value='time'>TIME</option>
    </select>
    Y-axis:
    <select name='yaxis' style="width:40px;">
      <option value='' selected>--option--</option>
      <option value='total'>TOTAL</option>
      <option value='fight'>FIGHT</option>
      <option value='car'>CAR</option>
      <option value='gun'>GUN</option>
    </select>
  DATE: <input type="text" id="date" placeholder="date" size="10" style="display: inline-block;" />
  <button id="btn" type="button" onclick="drawChart()">draw</button>

  <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
  <script type="text/javascript">
  google.charts.load('current', {'packages':['corechart']});
  google.charts.load('current', {'packages':['bar']});
  google.charts.setOnLoadCallback(drawChart);
    function drawChart(){
      var type=$('select[name=chart]').val();
      var x=$('select[name=x]').val();
      var y=$('select[name=yaxis]').val();
      var date = document.getElementById("date").value;
      if(type=='donut'){
        <?php
        $date=$_POST['date'];
        $sql="select count(*) as total, detectAction from detect_log group by detectAction";
        $result=mysqli_query($mysqli,$sql);
        $num=mysqli_num_rows($result);
         ?>
         var data = google.visualization.arrayToDataTable([
           ['ACCIDENT', 'Total'],
           <?php
           while($row=mysqli_fetch_array($result,MYSQLI_ASSOC)){
             echo "['".$row['detectAction']."',".$row['total']."],";
           }?>
         ]);
         var options = {
           title: 'ACCIDENT IN DB'
         };
         var chart = new google.visualization.PieChart(document.getElementById('piechart'));
         chart.draw(data, options);
       }
       if(type=='line'){
         if(x=='time'){
           if(y=='fight'){
             <?php
             $sql="select detectAction,logDate,count(*)as total,logTime from detect_log group by detectAction,logDate,logTime having logDate='2019-09-05'and detectAction='fight'";
             $result=mysqli_query($mysqli,$sql);
             $num=mysqli_num_rows($result);
             ?>
             var data = google.visualization.arrayToDataTable([
               ['Time', 'Fight'],
               <?php
               while($row=mysqli_fetch_array($result,MYSQLI_ASSOC)){
                 echo "['".$row['logTime']."',".$row['total']."],";
               }?>
             ]);
             var options = {
               title: 'A number of Fight',
               curveType: 'function',
               legend: { position: 'bottom' }
             };
             var chart = new google.visualization.LineChart(document.getElementById('piechart'));
             chart.draw(data, options);
           }
         }
       }
       if(type=='bar'){
         if(x=='area'){
           if(y=='total'){
             <?php
             $sql="select count(*) as total,screenPlace from detect_log group by screenPlace";
             $result=mysqli_query($mysqli,$sql);
             $num=mysqli_num_rows($result);
              ?>
              var data = google.visualization.arrayToDataTable([
                ['Area', y, { role: 'style' } ],
                <?php
                 while($row=mysqli_fetch_array($result,MYSQLI_ASSOC)){
                   echo "['".$row['screenPlace']."',".$row['total'].",'color: gray'],";
                 }?>
               ]);
               var options = {
                 title: 'Total number of Crime'
                };
                var chart = new google.charts.Bar(document.getElementById('piechart'));
                 chart.draw(data, google.charts.Bar.convertOptions(options));
           }
           if(y=='car'){
             <?php
             $sql="select count(*) as total,screenPlace from detect_log group by screenPlace,detectAction having detectAction='car'";
             $result=mysqli_query($mysqli,$sql);
             $num=mysqli_num_rows($result);
              ?>
              var data = google.visualization.arrayToDataTable([
                ['Area', y, { role: 'style' } ],
                <?php
                 while($row=mysqli_fetch_array($result,MYSQLI_ASSOC)){
                   echo "['".$row['screenPlace']."',".$row['total'].",'color: gray'],";
                 }?>
               ]);
               var options = {
                 title: 'Total number of '+y
                };
                var chart = new google.charts.Bar(document.getElementById('piechart'));
                 chart.draw(data, google.charts.Bar.convertOptions(options));
                  }
            if(y=='fight'){
              <?php
              $sql="select count(*) as total,screenPlace from detect_log group by screenPlace,detectAction having detectAction='fight'";
              $result=mysqli_query($mysqli,$sql);
              $num=mysqli_num_rows($result);
              ?>
              var data = google.visualization.arrayToDataTable([
                   ['Area', y, { role: 'style' } ],
                    <?php
                    while($row=mysqli_fetch_array($result,MYSQLI_ASSOC)){
                    echo "['".$row['screenPlace']."',".$row['total'].",'color: gray'],";
                    }?>
                   ]);
                    var options = {
                      title: 'Total number of '+y
                      };
                    var chart = new google.charts.Bar(document.getElementById('piechart'));
                    chart.draw(data, google.charts.Bar.convertOptions(options));
                   }
                else if(y=='gun'){
                  <?php
                   $sql="select count(*) as total,screenPlace from detect_log group by screenPlace,detectAction having detectAction='gun'";
                   $result=mysqli_query($mysqli,$sql);
                   $num=mysqli_num_rows($result);
                  ?>
                 var data = google.visualization.arrayToDataTable([
                    ['Area', y, { role: 'style' } ],
                    <?php
                     while($row=mysqli_fetch_array($result,MYSQLI_ASSOC)){
                     echo "['".$row['screenPlace']."',".$row['total'].",'color: gray'],";
                     }?>
                    ]);
                    var options = {
                      title: 'Total number of '+y
                      };
                    var chart = new google.charts.Bar(document.getElementById('piechart'));
                      chart.draw(data, google.charts.Bar.convertOptions(options));
                   }
                   else{
                     <?php
                      $sql="select screenPlace,count(*) as total,detectAction from detect_log group by detectAction,screenPlace having detectAction='fight'";
                      $sq="select screenPlace,count(*) as total,detectAction from detect_log group by detectAction,screenPlace having detectAction='car'";
                      $s="select screenPlace,count(*) as total,detectAction from detect_log group by detectAction,screenPlace having detectAction='gun'";
                      $result=mysqli_query($mysqli,$sql);
                      $resul=mysqli_query($mysqli,$sq);
                      $resu=mysqli_query($mysqli,$s);
                      $num=mysqli_num_rows($result);
                     ?>
                     var data = google.visualization.arrayToDataTable([
                       ['Area', 'Fight', 'Car', 'Gun'],
                       <?php
                       while($r=mysqli_fetch_array($resu,MYSQLI_ASSOC)){
                         $row=mysqli_fetch_array($result,MYSQLI_ASSOC);
                         $ro=mysqli_fetch_array($resul,MYSQLI_ASSOC);
                       echo "['".$r['screenPlace']."',".$row['total'].",".$ro['total'].",".$r['total']."],";
                       }?>
                     ]);
                     var options = {
                       chart: {
                         title: 'Area(Total: fight/car/gun)',
                       },
                     };
                     var chart = new google.charts.Bar(document.getElementById('piechart'));
                     chart.draw(data, google.charts.Bar.convertOptions(options));
                   }
                 }
               }
              }
  </script>
    <div id="piechart" style="width:550px;margin:5px;height:470px;border:3px solid #333;"></div>
  </div>
</section>

  <!--기록 DB와 연결되어 기간 검색 등 구현-->
  <section id="search">
<section id="search_section">
  <div id="side_right_bottom2">
    <!--날짜 검색 텍스트-->
    <form id=search_form>
      START: <input type="text" id="startDate" class="form-control" size="10"/>&nbsp &nbsp
      END: <input type="text" id="endDate" class="form-control" size="10" /> &nbsp&nbsp
      ACTION: <select name="action">
        <option value="none">NONE</option>
        <option value="fight">FIGHT</option>
        <option value="car">CARACCIDENT</option>
        <option value="gun">GUN</option>
      </select>
      <input type="button" value="search" id="search"/>
    </form>

    <!--jquery 날짜-->
    <div id = "response"></div>
    <span id="details"></span>
    <link rel="stylesheet" href="http://code.jquery.com/ui/1.8.18/themes/base/jquery-ui.css" type="text/css" />
    <!--jQuery 기본 js파일-->
    <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js"></script>
    <!--jQuery UI 라이브러리 js파일-->
    <script src="http://code.jquery.com/ui/1.8.18/jquery-ui.min.js"></script>

    <!--선택 부분-->
    <script>
    $(function(){
      $("#startDate").datepicker({
        changeMonth:true,
        changeYear: true,
        maxDate:0,
        dateFormat:"yy-mm-dd"
      });
    });
    $(function(){
      $("#endDate").datepicker({
        changeMonth:true,
        changeYear: true,
        maxDate:0,
        dateFormat:"yy-mm-dd"
      });
    });
    </script>

    <!--날짜 Db연동-->
    <script>
    var searchList;
    $('#search').click(function(){
      var startDate=$('#startDate').val();
      var endDate=$('#endDate').val();
      var action=$('select[name=action]').val();
      $('#search').prop('disabled',true);
      if(startDate=='' || endDate==''){
        $('#response').html("&nbsp&nbsp&nbsp<span class='text-danger'>All Fields are required</span>");
        $('#search').prop('disabled',false);
      }
      else{
        $.post(
          'tt.php',{startDate:startDate,endDate:endDate,action:action},
          function(data){
            searchList=JSON.parse(data);
            $('#response').empty();
            for(var i=0; i< searchList.length; i++){
               var item =searchList[i].detectAction +" "+searchList[i].logDate + " " + searchList[i].logTime +" Screen"+ searchList[i].screenNumber +" " + searchList[i].screenPlace;
               item = "<br/>&nbsp&nbsp<a href=\'javascript:openPop("+i+")\'>" + item + "</a>";
               $('#response').append(item);
             }
             $('#search').prop('disabled', false);
          }
        );
      }
    });
    function openPop(num){
      window.open(searchList[num].video,"top=100, left=50, width=400, height=200, scrollbars=no, resizable=no ,status=no ,toolbar=no");
 }

    </script>
  </div>
</section>
</select>
  <!--알림 선택 서비스 부분-->
  <!--
  <div id="side_left_bottom">
    <br><br>&nbsp&nbsp&nbsp
    <input type="checkbox" name="agree" value="BROADCAST"> BROADCAST
    <br><br>&nbsp&nbsp&nbsp
    <input type="checkbox" name="agree" value="MESSAGE SECURITY"> MESSAGE SECURITY
    <br><br>&nbsp&nbsp&nbsp
    <input type="checkbox" name="agree" value="CALL SECURITY" > CALL SECURITY
    <br><br>&nbsp&nbsp&nbsp
    <input type="button" value="alert start" onclick="goCheck()"/>
      <script type="text/javascript">
          function goCheck(){
            var s="/";
            var chkBox=document.getElementsByName("agree");
            for (var i=0;i<chkBox.length;i++){
              if(chkBox[i].checked){
                s+=chkBox[i].value;
                s+="/";
              }
            }
            if(s!="/"){
              alert("Start "+s);
            }
            else{
              alert("Please select a service");
            }
          }
        </script>
  </div>
-->


</section>
<!--IoT & Long Polling Part-->
<script src="https://www.gstatic.com/firebasejs/4.8.2/firebase.js"></script>
<script src="https://www.gstatic.com/firebasejs/4.8.1/firebase-firestore.js"></script>
<script type="text/javascript">
  

var eventManager = new function(){
    // Your web app's Firebase configuration
  var firebaseConfig = {
    apiKey: "AIzaSyBRwcoB8YxlJ1INyvMtEjzexGlqMIgDJEo",
    authDomain: "controlalert-44d81.firebaseapp.com",
    databaseURL: "https://controlalert-44d81.firebaseio.com",
    projectId: "controlalert-44d81",
    storageBucket: "controlalert-44d81.appspot.com",
    messagingSenderId: "115563763959",
    appId: "1:115563763959:web:815b4cadbfbc40bd641bf2",
    measurementId: "G-DESBNBTDC0"
  };
  // Initialize Firebase
  firebase.initializeApp(firebaseConfig);
  //firebase.analytics();

  var db =firebase.firestore();


  var idle    = true;
  var interval  = 500;
  var xmlHttp   = new XMLHttpRequest();
  var finalNo = '';

  // Ajax Setting
  xmlHttp.onreadystatechange = function()
  {
    if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
    {
      // JSON 포맷으로 Parsing
      res = JSON.parse(xmlHttp.responseText);
      finalNo = res.no;
      
      //내용 보여주기
    //  eventManager.show(res.data);
      eventManager.show(res.status);
      //내용 가져오기
      eventManager.proc();
    }
  }

  //내용 가져오기
  this.proc = function()
  {
    // Ajax 통신
    xmlHttp.open("GET", "proc.php?no=" + encodeURIComponent(finalNo), true);
    xmlHttp.send();
  }
/*
  // 내용 보여주기
  this.show = function(data)
  {
    var o = document.getElementById('head_alarm');
    var dt;
    
    //내용 추가
    for(var i=0; i<data.length; i++)
    {
      //o.();
      dt = document.createElement('dt');
      dt.appendChild(document.createTextNode(data[i].detectAction));
      //alert(dt.value);
      o.appendChild(dt);
      //o.empty;
    }

  }
*/

// 내용 보여주기
  this.show = function(statusLED)
  {
 

    db.collection("LEDstatus").doc("fn8vTw7P0Kds381EhwO2")
    .set({
      status:statusLED
    })
    .then(function(){
      console.log("Written Successfully");
    })
    .catch(function(error){
      console.error("error :: ", error);
    });
     
  }
setTimeout(function() {
      db.collection("LEDstatus").doc("fn8vTw7P0Kds381EhwO2")
      .set({
        status:false
      })
    }, 3000);
  
}

window.onload = function()
{
  eventManager.proc();
}
</script>
</body>
</html>

